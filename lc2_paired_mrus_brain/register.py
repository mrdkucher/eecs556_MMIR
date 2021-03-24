"""
Affine iterative pairwise registration via LC2
"""
import argparse
import os
import shutil
import sys

import tensorflow as tf
import pybobyqa
import numpy as np
import nibabel as nib

import deepreg.model.layer_util as layer_util
import deepreg.util as util
from deepreg.registry import REGISTRY

import scipy.interpolate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # no info, warnings printed
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--moving",
    help="File path to moving image",
    dest="moving",
    action="store"
)
parser.add_argument(
    "-f", "--fixed",
    help="File path to fixed image",
    dest="fixed",
    action="store"
)
parser.add_argument(
    "-t", "--tag",
    help="File path to landmark file: (.tag)",
    dest="tag",
    action="store"
)
parser.add_argument(
    "-lf", "--fixed-landmark",
    help="File path to fixed image landmarks (as spheres in nifti)",
    dest="fixed_landmark",
    action="store"
)
parser.add_argument(
    "-lm", "--moving-landmark",
    help="File path to moving image landmarks (as spheres in nifti)",
    dest="moving_landmark",
    action="store"
)
parser.add_argument(
    "-s", "--image_size",
    help="3-entry tuple to resize image e.g. (256, 256, 288)",
    dest="image_size",
    nargs=3,
    action="store",
    type=int,
    default=[-1, -1, -1]
)
parser.add_argument(
    "-nn", "--no_normalize",
    help="Do not normalize input US and MRI images to [0, 1] before registration (default is to normalize)",
    dest="no_normalize",
    action="store_true",
    default=False
)
parser.add_argument(
    "--max_iter",
    help="number of iterations to run",
    dest="max_iter",
    action="store",
    type=int,
    default=1000
)
parser.add_argument(
    "--verbose_bobyqa",
    help="use verbose output for bobyqa solver",
    dest="v_bobyqa",
    action="store_true",
    default=False
)
parser.add_argument(
    "-g", "--seek_global_minimum",
    help="enable seek global minimum option for bobyqa solver",
    dest="seek_global_minimum",
    action="store_true",
    default=False
)
args = parser.parse_args()

MAIN_PATH = os.getcwd()
PROJECT_DIR = sys.argv[0].split('register.py')[0]
if PROJECT_DIR == "":
    PROJECT_DIR = "./"
os.chdir(PROJECT_DIR)

MOVING_PATH = args.moving
FIXED_PATH = args.fixed
MOVING_LM_PATH = args.moving_landmark
FIXED_LM_PATH = args.fixed_landmark
TAG_PATH = args.tag

# registration parameters
image_loss_config = {"name": "lc2"}

use_landmarks = True
use_tags = True
# check for images and landmarks
if not os.path.exists(MOVING_PATH):
    raise FileNotFoundError(f"Moving image not found at: {MOVING_PATH}")
if not os.path.exists(FIXED_PATH):
    raise FileNotFoundError(f"Fixed image not found at: {FIXED_PATH}")
if not MOVING_LM_PATH or not os.path.exists(MOVING_LM_PATH):
    print(f"Moving landmarks not found at: {MOVING_LM_PATH}, not warping landmarks")
    use_landmarks = False
if not FIXED_LM_PATH or not os.path.exists(FIXED_LM_PATH):
    print(f"Fixed landmarks not found at: {FIXED_LM_PATH}, not warping landmarks")
    use_landmarks = False
if not TAG_PATH or not os.path.exists(TAG_PATH):
    print(f"Landmarks not found at: {TAG_PATH}, not calculating mTRE")
    use_tags = False


def load_preprocess_image(file_path, normalize=True, fixed=False, resize=True, method='bilinear'):
    """
    Load in nifti image, save as a tensor, resize image
    to image_size given in args, and (optionally) normalize.

    :param file_path: path to .nii.gz file (str)
    :param normalize: normalize image to [0, 1] (bool)
    :param method: image resize method. See tf.image.ResizeMethod (str)
    :return: image tensor (1, h, w, d)
    """
    nii = nib.load(file_path)
    image = nii.get_fdata()
    image_aff = nii.affine
    orig_shape = np.array(image.shape)

    image = tf.expand_dims(tf.convert_to_tensor(image, dtype=tf.float32), axis=0)

    # rescale fixed image
    if fixed:
        image = ((image + 150.0) / (1700.0 + 150.0)) * 255.0

    # resize to arg.image_size
    if args.image_size != [-1, -1, -1] and resize:
        image = layer_util.resize3d(image, args.image_size, method=method)
        scale = orig_shape / np.array(args.image_size)
        # convert to homogeneous coords
        scale = np.concatenate((scale, np.ones(1)))
        # apply scaling to image affine
        image_aff = image_aff @ np.diag(scale)

    if normalize and not args.no_normalize:
        # normalize to [0, 1]
        image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
        if fixed:
            image = image + 1e-10  # add constant so it's not all 0s

    return image, image_aff


# load and preprocess fixed and moving images
moving_image, moving_image_aff = load_preprocess_image(MOVING_PATH, normalize=True)  # normalized
fixed_image, fixed_image_aff = load_preprocess_image(FIXED_PATH, fixed=True, normalize=True)  # normalized

if use_landmarks:
    # load and prepreprocess fixed and moving landmarks (images)
    # resize wth 'nearest' interp: need integer values 1-15
    moving_lm_img, _ = load_preprocess_image(MOVING_LM_PATH, normalize=False, method='nearest')
    fixed_lm_img, _ = load_preprocess_image(FIXED_LM_PATH, normalize=False, method='nearest')
    if (np.all(np.unique(moving_lm_img) != np.unique(fixed_lm_img))):
        print("Warning: landmark files don't have same integer-valued landmark spheres after re-sampling")
if use_tags:
    # Adapted from landmarks_split_txt:
    # https://gist.github.com/mattiaspaul/56a49fa792ef6f143e56699a06067712
    landmarks = []
    with open(args.tag) as f:
        lines = f.readlines()
        for line in lines[5:]:  # skip first 5 lines
            line = line.strip("\n").strip(";").strip("\"").strip(" ")
            landmarks.append(np.fromstring(line, dtype=float, sep=' '))

    landmarks = np.asarray(landmarks)
    fixed_landmarks = landmarks[:, :3]
    moving_landmarks = landmarks[:, 3:]


# Get a reference grid (meshgrid) for warping
fixed_image_size = fixed_image.shape
grid_ref = layer_util.get_reference_grid(grid_size=fixed_image_size[1:4])

# Print config:
print("LC2 Config:")
print("Fixed Image:")
print("    path:", FIXED_PATH)
print("    size:", fixed_image.shape)
print("    landmarks path:", FIXED_LM_PATH)
print("    min:", tf.reduce_min(fixed_image))
print("    max:", tf.reduce_max(fixed_image))
print("Moving Image:")
print("    path:", MOVING_PATH)
print("    size:", moving_image.shape)
print("    landmarks path:", MOVING_LM_PATH)
print("    min:", tf.reduce_min(moving_image))
print("    max:", tf.reduce_max(moving_image))
print("Tag File:", args.tag)
print("max iterations:", args.max_iter)
print("normalize inputs:", not args.no_normalize)
print("seek global minimum:", args.seek_global_minimum)
print()


# BOBYQA optimization:
def build_objective_function(grid, mov, fix) -> object:
    """
    Builder function for the function which BOBYQA optimizes

    :param grid: reference grid return from layer_util.get_reference_grid
    :param weights: trainable affine parameters [4, 3]
    :param mov: moving image [1, m_dim1, m_dim2, m_dim3]
    :param fix: fixed image [1, f_dim1, f_dim2, f_dim3]
    :return loss: image dissimilarity
    """
    loss_fn = REGISTRY.build_loss(config=image_loss_config)

    def objective_function(weights) -> object:
        """
        Objective function for BOBYQA to minimize:
        Performs affine transformation on mov, defined by weights:

        :param weights: affine transformation, shape [1, 4, 3],
            transpose of normal affine matrix:
            y_pred = [y_true | 1] * weights
            weights = [[a11, a21, a31],
                       [a12, a22, a32],
                       [a13, a23, a33],
                       [a14, a24, a34]]
            where a14, a24, and a34 correspond to the bias weights
            for x, y, and z, accordingly

        :return: image dissimilarity measure, to minimize

        """
        weights = tf.convert_to_tensor(
            weights.reshape((1, 4, 3)), dtype=tf.float32)
        pred = layer_util.resample(
            vol=mov, loc=layer_util.warp_grid(grid, weights))
        return loss_fn(y_true=fix, y_pred=pred)

    return objective_function


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

obj_fn = build_objective_function(grid_ref, moving_image, fixed_image)
soln = pybobyqa.solve(obj_fn, var_affine, bounds=(lower, upper), rhobeg=0.1,
                      print_progress=args.v_bobyqa, maxfun=args.max_iter,
                      seek_global_minimum=args.seek_global_minimum)
print(soln)

var_affine = tf.convert_to_tensor(soln.x.reshape((1, 4, 3)), dtype=tf.float32)

# warp the moving image using the optimized affine transformation
grid_opt = layer_util.warp_grid(grid_ref, var_affine)
warped_moving_image = layer_util.resample(vol=moving_image, loc=grid_opt)
if use_landmarks:
    # # warp the moving landmarks, too
    # warped_moving_lm_img = layer_util.resample(
    #     vol=moving_lm_img, loc=grid_opt)

    # use nearest neighbor interpolation so landmarks can be properly extracted later
    grid_ref_np = grid_ref.numpy().reshape((-1, 3))
    moving_lm_img_np = moving_lm_img.numpy().squeeze().reshape(-1)
    grid_opt_np = grid_opt.numpy().squeeze().reshape((-1, 3))
    warped_moving_lm_img = scipy.interpolate.griddata(grid_ref_np, moving_lm_img_np, grid_opt_np, method='nearest')
    warped_moving_lm_img = warped_moving_lm_img.reshape(moving_lm_img.shape)
    warped_moving_lm_img = tf.convert_to_tensor(warped_moving_lm_img, dtype=tf.float32)


def calculate_mTRE(xyz_true, xyz_predict):
    assert xyz_true.shape == xyz_predict.shape
    TRE = np.sqrt(np.sum(np.power(xyz_true - xyz_predict, 2), axis=1))
    mTRE = np.mean(TRE)
    return mTRE


def extract_centroid(image):
    """
    Extract centroid from nifti images with landmark spheres
    which have integer values according to labels
    Adapted from: https://gist.github.com/mattiaspaul/f4183f525b1cbc65e71ad23298d6436e

    :param image:
        - shape: (dim_1, dim_2, dim_3) or (batch, dim_1, dim_2, dim_3)
        - tensor or numpy array
    
    :return positions:
        - numpy array of labels 1
    """
    if tf.is_tensor(image):
        image = image.numpy()
    if len(image.shape) == 4:
        image = image.squeeze()
    assert len(image.shape) == 3

    x = np.linspace(0, image.shape[0] - 1, image.shape[0])
    y = np.linspace(0, image.shape[1] - 1, image.shape[1])
    z = np.linspace(0, image.shape[2] - 1, image.shape[2])
    yv, xv, zv = np.meshgrid(y, x, z)
    unique = np.unique(image)[1:]  # don't include 0
    positions = np.zeros((len(unique), 3))
    for i in range(len(unique)):
        label = (image == unique[i]).astype('float32')
        xc = np.sum(label * xv) / np.sum(label)
        yc = np.sum(label * yv) / np.sum(label)
        zc = np.sum(label * zv) / np.sum(label)
        positions[i, 0] = xc
        positions[i, 1] = yc
        positions[i, 2] = zc
    return positions


# Calculate mTRE (in world coords)
if use_tags:
    num_lms = moving_landmarks.shape[0]
    bias = np.ones((num_lms, 1))
    # Landmark - based:
    # Landmarks -> [inverse moving_image affine] -> landmarks in pixels -> [affine xform] -> warped landmarks in pixels -> [fixed_image affine] -> warped moving landmarks
    # convert to homogeneous coordinates
    mov_hlms = np.concatenate((moving_landmarks, bias), axis=1)
    # convert to pixel values from world coords
    mov_hpix = mov_hlms @ np.linalg.inv(moving_image_aff).T
    # perform transformation
    aff_xform = var_affine.numpy().squeeze()
    warped_moving_pixels = mov_hpix @ aff_xform
    # transform back to world coords
    warped_moving_pixels_world = np.concatenate((warped_moving_pixels, bias), axis=1) @ fixed_image_aff.T
    warped_moving_pixels_world = warped_moving_pixels_world[:, :3]
    mTRE1 = calculate_mTRE(fixed_landmarks, warped_moving_pixels_world)
    print("landmark mTRE:", mTRE1)

    # Sphere - based:
    # Extract warped centroids, extract fixed centroids, convert to world coords
    warped_moving_lm_centroid_pixels = extract_centroid(warped_moving_lm_img)
    warped_moving_lm_centroids = np.concatenate((warped_moving_lm_centroid_pixels, bias), axis=1) @ fixed_image_aff.T
    warped_moving_lm_centroids = warped_moving_lm_centroids[:, :3]
    fixed_lm_centroid_pixels = extract_centroid(fixed_lm_img)
    fixed_lm_centroids = np.concatenate((fixed_lm_centroid_pixels, bias), axis=1) @ fixed_image_aff.T
    fixed_lm_centroids = fixed_lm_centroids[:, :3]
    # pixel_mTRE = calculate_mTRE(fixed_lm_centroid_pixels, warped_moving_lm_centroid_pixels)
    # print("pixel mTRE:", pixel_mTRE)
    mTRE4 = calculate_mTRE(fixed_lm_centroids, warped_moving_lm_centroids)
    print("sphere mTRE:", mTRE4)

# save output to files
SAVE_PATH = "logs_reg"
if os.path.exists(SAVE_PATH):
    shutil.rmtree(SAVE_PATH)
os.mkdir(SAVE_PATH)

# Save affine transformation:
np.savetxt(os.path.join(SAVE_PATH, "affine_T.txt"), aff_xform)

arrays = [
    tf.transpose(a, [1, 2, 3, 0]) if a.ndim == 4 else tf.squeeze(a)
    for a in [
        moving_image,
        fixed_image,
        warped_moving_image,
    ]
]
arr_names = [
    "moving_image",
    "fixed_image",
    "warped_moving_image",
]
for arr, arr_name in zip(arrays, arr_names):
    for n in range(arr.shape[-1]):
        util.save_array(
            save_dir=SAVE_PATH,
            arr=arr[..., n],
            name=arr_name + (arr.shape[-1] > 1) * "_{}".format(n),
            # label's value is already in [0, 1]
            normalize="image" in arr_name,
        )
if use_landmarks:
    arrays = [
        tf.transpose(a, [1, 2, 3, 0]) if a.ndim == 4 else tf.squeeze(a)
        for a in [
            moving_lm_img,
            fixed_lm_img,
            warped_moving_lm_img,
        ]
    ]
    arr_names = [
        "moving_landmarks",
        "fixed_landmarks",
        "warped_moving_landmarks",
    ]
    for arr, arr_name in zip(arrays, arr_names):
        for n in range(arr.shape[-1]):
            util.save_array(
                save_dir=SAVE_PATH,
                arr=arr[..., n],
                name=arr_name + (arr.shape[-1] > 1) * "_{}".format(n),
                # label's value is already in [0, 1]
                normalize="image" in arr_name,
            )
os.chdir(MAIN_PATH)
