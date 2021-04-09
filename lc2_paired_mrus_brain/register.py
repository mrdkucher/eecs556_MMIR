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
import numpy as np

import deepreg.model.layer_util as layer_util
import deepreg.util as util
from lc2_util import load_preprocess_image, build_objective_function, calculate_mTRE, extract_centroid

import scipy.interpolate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # no info, warnings printed


def main(args):
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
    print('seek global minimum:', args.seek_global_minimum)
    print('output folder:', args.output)
    print('Use patch:', args.patch)
    print('Patch size:', args.patch_size)
    print('Use neighborhood:', args.neighborhood)
    print()

    # affine transformation as trainable weights
    var_affine = np.array(
        [1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0,
         0.0, 0.0, 0.0], dtype=np.float32)
    lower = np.array(
        [-10.0, -10.0, -10.0,
         -10.0, -10.0, -10.0,
         -10.0, -10.0, -10.0,
         -100.0, -100.0, -100.0], dtype=np.float32)
    upper = np.array(
        [10.0, 10.0, 10.0,
         10.0, 10.0, 10.0,
         10.0, 10.0, 10.0,
         100.0, 100.0, 100.0], dtype=np.float32)

    # DAK TODO: 2-step optimization (2018 Team ImFusion)
    #   1) TODO: a global DIRECT (DIviding RECTangles) sub-division method searches on translation only
    #   2) Done: a local BOBYQA algorithm on all six(???) rigid transformation parameters (3 rotations + 3 translations?)
    start_time = time.time()
    lc2_loss_config = {'name': 'lc2', 'patch': args.patch, 'patch_size': args.patch_size, 'neighborhood': args.neighborhood}
    obj_fn = build_objective_function(grid_ref, moving_image, fixed_image,
                                      image_loss_config=lc2_loss_config)
    soln = pybobyqa.solve(obj_fn, var_affine, bounds=(lower, upper), rhobeg=0.1,
                          print_progress=args.v_bobyqa, maxfun=args.max_iter,
                          seek_global_minimum=args.seek_global_minimum)
    print(soln)
    end_time = time.time() - start_time

    aff_xform_T = soln.x.reshape((4, 3))
    var_affine = tf.convert_to_tensor(
        soln.x.reshape((1, 4, 3)), dtype=tf.float32)

    # warp the moving image using the optimized affine transformation
    grid_opt = layer_util.warp_grid(grid_ref, var_affine)
    warped_moving_image = layer_util.resample(vol=moving_image, loc=grid_opt)
    if use_labels:
        # # warp the moving landmarks, too
        # warped_moving_lm_img = layer_util.resample(
        #     vol=moving_lm_img, loc=grid_opt)

        # use nearest neighbor interpolation so landmarks can be properly extracted later
        grid_ref_np = grid_ref.numpy().reshape((-1, 3))
        moving_label_np = moving_label.numpy().squeeze().reshape(-1)
        grid_opt_np = grid_opt.numpy().squeeze().reshape((-1, 3))
        warped_moving_label = scipy.interpolate.griddata(
            grid_ref_np, moving_label_np, grid_opt_np, method='nearest')
        warped_moving_label = warped_moving_label.reshape(moving_label.shape)
        warped_moving_label = tf.convert_to_tensor(
            warped_moving_label, dtype=tf.float32)

    # Calculate mTRE (in world coords)
    if use_tags:
        num_lms = moving_landmarks.shape[0]
        bias = np.ones((num_lms, 1))
        # Landmark - based:
        # Landmarks -> [inverse moving_image affine] -> landmarks in voxels -> [affine xform] ->
        # warped landmarks in voxels -> [fixed_image affine] -> warped moving landmarks
        # convert to homogeneous coordinates
        mov_hlms = np.concatenate((moving_landmarks, bias), axis=1)
        # convert to pixel values from world coords
        mov_hvox = mov_hlms @ np.linalg.inv(moving_image_aff).T
        # perform transformation
        warped_moving_voxels = mov_hvox @ aff_xform_T
        # transform back to world coords
        warped_moving_landmarks = np.concatenate(
            (warped_moving_voxels, bias), axis=1) @ fixed_image_aff.T
        warped_moving_landmarks = warped_moving_landmarks[:, :3]
        mTRE = calculate_mTRE(fixed_landmarks, warped_moving_landmarks)
        print('landmark mTRE:', mTRE)

    if use_labels:
        # Sphere - based:
        # Extract warped centroids, extract fixed centroids, convert to world coords
        warped_moving_label_centroid_voxels = extract_centroid(
            warped_moving_label)
        warped_moving_label_centroids = np.concatenate(
            (warped_moving_label_centroid_voxels, bias), axis=1) @ fixed_image_aff.T
        warped_moving_label_centroids = warped_moving_label_centroids[:, :3]
        fixed_label_centroid_voxels = extract_centroid(fixed_label)
        fixed_label_centroids = np.concatenate(
            (fixed_label_centroid_voxels, bias), axis=1) @ fixed_image_aff.T
        fixed_label_centroids = fixed_label_centroids[:, :3]
        voxel_mTRE = calculate_mTRE(
            fixed_label_centroid_voxels, warped_moving_label_centroid_voxels)
        print('voxel mTRE:', voxel_mTRE)
        mTRE4 = calculate_mTRE(fixed_label_centroids,
                               warped_moving_label_centroids)
        print('sphere mTRE:', mTRE4)

    # save output to files
    SAVE_PATH = args.output
    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
    os.mkdir(SAVE_PATH)

    # Save registration results
    with open(os.path.join(SAVE_PATH, 'reg_results.txt'), 'w') as f:
        f.write('Finished alignment in {:f} seconds\n'.format(end_time))
        if use_tags:
            f.write('landmark mTRE: {:f}\n'.format(mTRE))
        if use_labels:
            f.write('voxel mTRE: {:f}\n'.format(voxel_mTRE))
            f.write('sphere mTRE: {:f}\n'.format(mTRE4))
        f.write('\n')
        f.write(str(soln))

    arrays = [
        tf.transpose(a, [1, 2, 3, 0]) if a.ndim == 4 else tf.squeeze(a)
        for a in [
            moving_image,
            fixed_image,
            warped_moving_image,
        ]
    ]
    arr_names = [
        'moving_image',
        'fixed_image',
        'warped_moving_image',
    ]
    for arr, arr_name in zip(arrays, arr_names):
        for n in range(arr.shape[-1]):
            util.save_array(
                save_dir=SAVE_PATH,
                arr=arr[..., n],
                name=arr_name + (arr.shape[-1] > 1) * '_{}'.format(n),
                # label's value is already in [0, 1]
                normalize='image' in arr_name,
            )
    if use_labels:
        arrays = [
            tf.transpose(a, [1, 2, 3, 0]) if a.ndim == 4 else tf.squeeze(a)
            for a in [
                moving_label,
                fixed_label,
                warped_moving_label,
            ]
        ]
        arr_names = [
            'moving_label',
            'fixed_label',
            'warped_moving_label',
        ]
        for arr, arr_name in zip(arrays, arr_names):
            for n in range(arr.shape[-1]):
                util.save_array(
                    save_dir=SAVE_PATH,
                    arr=arr[..., n],
                    name=arr_name + (arr.shape[-1] > 1) * '_{}'.format(n),
                    # label's value is already in [0, 1]
                    normalize='image' in arr_name,
                )
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
        '-o', '--output',
        help='Output directory (defaults to logs_reg)',
        dest='output',
        action='store',
        default='logs_reg'
    )
    args = parser.parse_args()
    main(args)
