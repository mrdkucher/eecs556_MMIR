'''
Affine iterative pairwise registration via LC2
'''
import argparse
import os
import shutil
import sys
import time
import tensorflow as tf
import pybobyqa
from scipydirect import minimize
import numpy as np

import deepreg.model.layer_util as layer_util
from lc2_util import (load_preprocess_image, build_objective_function, warp_landmarks, labels2world_coords,
                      calculate_mTRE, create_transformation_mat, save_image, warp_image)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # no info, warnings printed


def reg(args):
    MAIN_PATH = os.getcwd()
    PROJECT_DIR = os.path.dirname(sys.argv[0])
    if PROJECT_DIR == '':
        PROJECT_DIR = './'
    os.chdir(PROJECT_DIR)

    MOVING_PATH = args.moving
    FIXED_PATH = args.fixed
    MOVING_LABEL_PATH = args.moving_label
    FIXED_LABEL_PATH = args.fixed_label
    TAG_PATH = args.tag

    use_labels = True
    use_tags = True
    # check for images and landmarks
    if not os.path.exists(MOVING_PATH):
        raise FileNotFoundError(f'Moving image not found at: {MOVING_PATH}')
    if not os.path.exists(FIXED_PATH):
        raise FileNotFoundError(f'Fixed image not found at: {FIXED_PATH}')
    if not MOVING_LABEL_PATH or not os.path.exists(MOVING_LABEL_PATH):
        print(
            f'Moving labels not found at: {MOVING_LABEL_PATH}, not warping labels')
        use_labels = False
    if not FIXED_LABEL_PATH or not os.path.exists(FIXED_LABEL_PATH):
        print(
            f'Fixed labels not found at: {FIXED_LABEL_PATH}, not warping labels')
        use_labels = False
    if not TAG_PATH or not os.path.exists(TAG_PATH):
        print(f'Landmarks not found at: {TAG_PATH}, not calculating mTRE')
        use_tags = False

    # load and preprocess fixed and moving images
    moving_image, moving_image_aff = load_preprocess_image(
        MOVING_PATH, image_size=args.image_size)
    fixed_image, fixed_image_aff = load_preprocess_image(
        FIXED_PATH, image_size=args.image_size, fixed=True)

    if use_labels:
        # load and prepreprocess fixed and moving landmarks (images)
        # resize wth 'nearest' interp: need integer values 1-15
        moving_label, _ = load_preprocess_image(MOVING_LABEL_PATH, image_size=args.image_size,
                                                normalize=False, method='nearest')
        fixed_label, _ = load_preprocess_image(FIXED_LABEL_PATH, image_size=args.image_size,
                                               normalize=False, method='nearest')
        if (np.all(np.unique(moving_label) != np.unique(fixed_label))):
            print(
                'Warning: label files don\'t have same integer-valued label spheres after re-sampling')
    if use_tags:
        # Adapted from landmarks_split_txt:
        # https://gist.github.com/mattiaspaul/56a49fa792ef6f143e56699a06067712
        landmarks = []
        with open(args.tag) as f:
            lines = f.readlines()
            for line in lines[5:]:  # skip first 5 lines
                line = line.strip('\n').strip(';').strip('\'').strip(' ')
                landmarks.append(np.fromstring(line, dtype=float, sep=' '))

        landmarks = np.asarray(landmarks)
        fixed_landmarks = landmarks[:, :3]
        moving_landmarks = landmarks[:, 3:]

    # Get a reference grid (meshgrid) for warping
    fixed_image_size = fixed_image.shape
    grid_ref = layer_util.get_reference_grid(grid_size=fixed_image_size[1:4])

    # Print config:
    print('LC2 Config:')
    print('Fixed Image:')
    print('    path:', FIXED_PATH)
    print('    size:', fixed_image.shape)
    print('    label path:', FIXED_LABEL_PATH)
    print('    min:', tf.reduce_min(fixed_image))
    print('    max:', tf.reduce_max(fixed_image))
    print('Moving Image:')
    print('    path:', MOVING_PATH)
    print('    size:', moving_image.shape)
    print('    label path:', MOVING_LABEL_PATH)
    print('    min:', tf.reduce_min(moving_image))
    print('    max:', tf.reduce_max(moving_image))
    print('Tag File:', args.tag)
    print('max iterations:', args.max_iter)
    print('Perform translation registration (DIRECT):', args.direct)
    print('Registration mode (BOBYQA):', args.reg_mode)
    print('Seek global minimum:', args.seek_global_minimum)
    print('Output folder:', args.output)
    print('Similarity measure:', args.similarity)
    if args.similarity == 'lc2':
        print("Using LC2 similarity measure")
        print('Use patch:', args.patch)
        print('Patch size:', args.patch_size)
        print('Use neighborhood:', args.neighborhood)

    start_time = time.time()
    # Define LC2 Config:
    if args.similarity == 'lc2':
        loss_config = {'name': 'lc2', 'patch': args.patch,
                       'patch_size': args.patch_size, 'neighborhood': args.neighborhood}
    else:
        loss_config = {'name': args.similarity}

    # 1) DIRECT on translation only:
    pred_translation = np.zeros(3, dtype=np.float32)
    if args.direct:
        print('Performing DIRECT translation optimization')
        translation_bounds = [(-50, 50), (-50, 50), (-50, 50)]
        obj_fun = build_objective_function(grid_ref, moving_image, fixed_image,
                                           image_loss_config=loss_config,
                                           transformation_type='translate')
        # use Jones modification
        direct_res = minimize(obj_fun, translation_bounds,
                              algmethod=1, maxf=args.max_iter)
        print(direct_res)
        print('\n')
        pred_translation = direct_res.x

    # 2) BOBYQA on 6 rigid (Rx, Ry, Rz, Tx, Ty, Tz)
    if args.reg_mode in ['rigid', 'both']:
        print('Performing BOBYQA rigid registration')
        var_rigid = np.array(
            [0.0, 0.0, 0.0,
             0.0, 0.0, 0.0], dtype=np.float32)
        var_rigid[3:] = pred_translation
        lower_bound = np.array(
            [-2.0,  -2.0,  -2.0,  # noqa: E201, E241
             -50.0, -50.0, -50.0], dtype=np.float32)  # noqa: E128, E201, E241
        upper_bound = lower_bound * -1.0
        lower_bound[3:] += pred_translation
        upper_bound[3:] += pred_translation
        obj_fun_rigid = build_objective_function(grid_ref, moving_image, fixed_image,
                                                 image_loss_config=loss_config,
                                                 transformation_type='rigid')
        soln_rigid = pybobyqa.solve(obj_fun_rigid, var_rigid, bounds=(lower_bound, upper_bound),
                                    print_progress=args.v_bobyqa, maxfun=args.max_iter, rhobeg=1.0,  # coarse
                                    seek_global_minimum=args.seek_global_minimum)
        print(soln_rigid)
        rigid_xform_T = create_transformation_mat(
            soln_rigid.x, transformation_type='rigid')
        var_rigid = tf.convert_to_tensor(
            rigid_xform_T.reshape((1, 4, 3)), dtype=tf.float32)
        warped_moving_image_rigid = warp_image(
            moving_image, grid_ref, var_rigid)
        if use_labels:
            warped_moving_label_rigid = warp_image(
                moving_label, grid_ref, var_rigid, method='nearest')
    else:  # must be doing affine registration, set initialization as identity
        rigid_xform_T = np.eye(4, 3, dtype=np.float32)

    # 3) BOBYQA on 12 affine
    if args.reg_mode in ['affine', 'both']:
        print('Performing BOBYQA affine registration')
        # get rigid transformation as affine initializer
        var_affine = rigid_xform_T.reshape(-1)
        lower_bound = np.array(
            [-10.0, -10.0, -10.0,
             -10.0, -10.0, -10.0,
             -10.0, -10.0, -10.0,
             -50.0, -50.0, -50.0], dtype=np.float32)
        upper_bound = lower_bound * -1.0
        lower_bound += var_affine
        upper_bound += var_affine
        obj_fn = build_objective_function(grid_ref, moving_image, fixed_image,
                                          image_loss_config=loss_config)
        soln = pybobyqa.solve(obj_fn, var_affine, bounds=(lower_bound, upper_bound), rhobeg=0.1,  # fine
                              print_progress=args.v_bobyqa, maxfun=args.max_iter,
                              seek_global_minimum=args.seek_global_minimum)
        print(soln)
        aff_xform_T = soln.x.reshape((4, 3))
        var_affine = tf.convert_to_tensor(
            soln.x.reshape((1, 4, 3)), dtype=tf.float32)
        warped_moving_image_affine = warp_image(
            moving_image, grid_ref, var_affine)
        if use_labels:
            warped_moving_label_affine = warp_image(
                moving_label, grid_ref, var_affine, method='nearest')

    end_time = time.time() - start_time

    # Calculate mTRE (in world coords)
    if use_tags:
        start_mTRE = calculate_mTRE(fixed_landmarks, moving_landmarks)
        print('starting mTRE:', start_mTRE)
        if args.reg_mode in ['rigid', 'both']:
            warped_moving_landmarks_rigid = warp_landmarks(
                moving_landmarks, moving_image_aff, rigid_xform_T, fixed_image_aff)
            rigid_mTRE = calculate_mTRE(
                fixed_landmarks, warped_moving_landmarks_rigid)
            print('landmark mTRE (rigid):', rigid_mTRE)
        if args.reg_mode in ['affine', 'both']:
            warped_moving_landmarks = warp_landmarks(
                moving_landmarks, moving_image_aff, aff_xform_T, fixed_image_aff)
            affine_mTRE = calculate_mTRE(
                fixed_landmarks, warped_moving_landmarks)
            print('landmark mTRE (affine):', affine_mTRE)

    if use_labels:
        if args.reg_mode in ['rigid', 'both']:
            moving_world, warped_moving_rigid_world, fixed_world = labels2world_coords(
                moving_label, warped_moving_label_rigid, fixed_label, fixed_image_aff)
            start_sphere_mTRE = calculate_mTRE(fixed_world, moving_world)
            rigid_sphere_mTRE = calculate_mTRE(fixed_world, warped_moving_rigid_world)
            print('sphere mTRE (rigid):', rigid_sphere_mTRE)
        if args.reg_mode in ['affine', 'both']:
            moving_world, warped_moving_affine_world, fixed_world = labels2world_coords(
                moving_label, warped_moving_label_affine, fixed_label, fixed_image_aff)
            start_sphere_mTRE = calculate_mTRE(fixed_world, moving_world)
            affine_sphere_mTRE = calculate_mTRE(fixed_world, warped_moving_affine_world)
            print('sphere mTRE (affine):', rigid_sphere_mTRE)

    # save output to files
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.mkdir(args.output)

    # Save registration results
    with open(os.path.join(args.output, 'reg_results.txt'), 'w') as f:
        f.write('Finished alignment in {:f} seconds\n'.format(end_time))
        # Print landmark mTRE
        if use_tags:
            f.write('start landmark mTRE: {:f}\n'.format(start_mTRE))
            if args.reg_mode in ['rigid', 'both']:
                f.write('landmark mTRE (rigid): {:f}\n'.format(rigid_mTRE))
            if args.reg_mode in ['affine', 'both']:
                f.write('landmark mTRE (affine): {:f}\n'.format(affine_mTRE))
        # Print label mTRE
        if use_labels:
            f.write('start sphere mTRE: {:f}\n'.format(start_sphere_mTRE))
            if args.reg_mode in ['rigid', 'both']:
                f.write('sphere mTRE (rigid): {:f}\n'.format(rigid_sphere_mTRE))
            if args.reg_mode in ['affine', 'both']:
                f.write('sphere mTRE (affine): {:f}\n'.format(affine_sphere_mTRE))
        f.write('\n')
        # Write BOBYQA Solution
        if args.reg_mode in ['rigid', 'both']:
            f.write('Rigid solution:\n')
            f.write(str(soln_rigid))
            f.write('\n')
        if args.reg_mode in ['affine', 'both']:
            f.write('Affine solution:\n')
            f.write(str(soln))

    # Save images
    images = [moving_image, fixed_image]
    names = ['moving_image', 'fixed_image']
    if args.reg_mode in ['rigid', 'both']:
        images.append(warped_moving_image_rigid)
        names.append('warped_moving_image_rigid')
    if args.reg_mode in ['affine', 'both']:
        images.append(warped_moving_image_affine)
        names.append('warped_moving_image_affine')
    if use_labels:
        images.append(moving_label)
        images.append(fixed_label)
        names.append('moving_label')
        names.append('fixed_label')
        if args.reg_mode in ['rigid', 'both']:
            images.append(warped_moving_label_rigid)
            names.append('warped_moving_label_rigid')
        if args.reg_mode in ['affine', 'both']:
            images.append(warped_moving_label_affine)
            names.append('warped_moving_label_affine')
    for (image, name) in zip(images, names):
        save_image(image, name, args.output)
    print('\n')
    os.chdir(MAIN_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--moving',
        help='File path to moving image',
        dest='moving',
        action='store',
        required=True
    )
    parser.add_argument(
        '-f', '--fixed',
        help='File path to fixed image',
        dest='fixed',
        action='store',
        required=True
    )
    parser.add_argument(
        '-t', '--tag',
        help='File path to landmark file: (.tag)',
        dest='tag',
        action='store'
    )
    parser.add_argument(
        '-lf', '--fixed-label',
        help='File path to fixed image labels (as spheres in nifti)',
        dest='fixed_label',
        action='store'
    )
    parser.add_argument(
        '-lm', '--moving-label',
        help='File path to moving image labels (as spheres in nifti)',
        dest='moving_label',
        action='store'
    )
    parser.add_argument(
        '-s', '--image-size',
        help='3-entry tuple to resize image e.g. (256, 256, 288)',
        dest='image_size',
        nargs=3,
        action='store',
        type=int,
        default=[-1, -1, -1]
    )
    parser.add_argument(
        '--similarity',
        help='Which similarity measure to use. Default == lc2',
        dest='similarity',
        choices=['lc2', 'gmi', 'lncc', 'ssd'],
        action='store',
        default='lc2'
    )
    parser.add_argument(
        '--no-patch',
        help='Use patch-based LC2',
        dest='patch',
        action='store_false',
    )
    parser.add_argument(
        '-p', '--patch-size',
        help='LC2 patch size. Default = 7',
        dest='patch_size',
        action='store',
        type=int,
        default=7
    )
    parser.add_argument(
        '-n', '--neighborhood',
        help='Use neighborhood intensity/gradient for LC2. Default = False',
        dest='neighborhood',
        action='store_true'
    )
    parser.add_argument(
        '--max-iter',
        help='number of iterations to run',
        dest='max_iter',
        action='store',
        type=int,
        default=1000
    )
    parser.add_argument(
        '--verbose-bobyqa',
        help='use verbose output for bobyqa solver',
        dest='v_bobyqa',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '-g', '--seek_global_minimum',
        help='enable seek global minimum option for bobyqa solver',
        dest='seek_global_minimum',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '-r', '--registration-mode',
        help='Registration mode',
        dest='reg_mode',
        action='store',
        choices=['rigid', 'affine', 'both'],
        default='affine'
    )
    parser.add_argument(
        '-d', '--direct',
        help='Perform intial translation registration via DIRECT',
        dest='direct',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '-o', '--output',
        help='Output directory (defaults to logs_reg)',
        dest='output',
        action='store',
        default='logs_reg'
    )
    args = parser.parse_args()
    reg(args)
