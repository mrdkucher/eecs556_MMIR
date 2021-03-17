"""
Affine iterative pairwise registration via LC2
"""
import argparse
import os
import shutil
import sys

import tensorflow as tf

import deepreg.model.layer_util as layer_util
import deepreg.util as util
from deepreg.registry import REGISTRY
from deepreg.dataset.loader.nifti_loader import load_nifti_file

import pybobyqa
import numpy as np

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

# registration parameters
image_loss_config = {"name": "lc2"}
learning_rate = 1e-3

use_landmarks = True
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


def load_preprocess_image(file_path, normalize=True, fixed=False):
    """
    Load in nifti image, save as a tensor, resize image
    to image_size given in args, and (optionally) normalize.

    :param file_path: path to .nii.gz file (str)
    :param normalize: normalize image to [0, 1] (bool)
    :return: image tensor (1, h, w, d)
    """
    image = load_nifti_file(file_path)
    image = tf.expand_dims(tf.convert_to_tensor(image), axis=0)

    # rescale fixed image
    if fixed:
        image = ((image + 150.0) / (1700.0 + 150.0)) * 255.0

    # resize to arg.image_size
    if args.image_size != [-1, -1, -1]:
        image = layer_util.resize3d(image, args.image_size)

    if normalize and not args.no_normalize:
        # normalize to [0, 1]
        image = (image - tf.reduce_min(image)) / \
            (tf.reduce_max(image) - tf.reduce_min(image))
        if fixed:
            image = image + 1e-10  # add constant so it's not all 0s

    return image


# load and preprocess fixed and moving images
moving_image = load_preprocess_image(MOVING_PATH, normalize=True)  # normalized
fixed_image = load_preprocess_image(
    FIXED_PATH, normalize=True, fixed=True)  # normalized

# load and prepreprocess fixed and moving landmarks (images)
if use_landmarks:
    moving_landmarks = load_preprocess_image(MOVING_LM_PATH, normalize=False)
    fixed_landmarks = load_preprocess_image(FIXED_LM_PATH, normalize=False)

# Get a reference grid for warping
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
print("max iterations:", args.max_iter)
print("normalize inputs:", not args.no_normalize)
print("seek global minimum:", args.seek_global_minimum)


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
        weights = tf.convert_to_tensor(weights.reshape((1, 4, 3)), dtype=tf.float32)
        pred = layer_util.resample(vol=mov, loc=layer_util.warp_grid(grid, weights))
        return loss_fn(y_true=fix, y_pred=pred)

    return objective_function


# affine transformation as trainable weights
var_affine = np.array(
    [0.0, 1.0, 0.0,
     -1.0, 0.0, 0.0,
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
     100.0, 100.0, 100.0])
obj_fn = build_objective_function(grid_ref, moving_image, fixed_image)
soln = pybobyqa.solve(obj_fn, var_affine, bounds=(lower, upper), rhobeg=1,
                      print_progress=args.v_bobyqa, maxfun=args.max_iter,
                      seek_global_minimum=args.seek_global_minimum)
print(soln)

var_affine = tf.convert_to_tensor(soln.x.reshape((1, 4, 3)), dtype=tf.float32)

# warp the moving image using the optimized affine transformation
grid_opt = layer_util.warp_grid(grid_ref, var_affine)
warped_moving_image = layer_util.resample(vol=moving_image, loc=grid_opt)
if use_landmarks:
    # warp the moving labels, too
    warped_moving_landmarks = layer_util.resample(vol=moving_landmarks, loc=grid_opt)


# save output to files
SAVE_PATH = "logs_reg"
if os.path.exists(SAVE_PATH):
    shutil.rmtree(SAVE_PATH)
os.mkdir(SAVE_PATH)

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
            normalize="image" in arr_name,  # label's value is already in [0, 1]
        )
if use_landmarks:
    arrays = [
        tf.transpose(a, [1, 2, 3, 0]) if a.ndim == 4 else tf.squeeze(a)
        for a in [
            moving_landmarks,
            fixed_landmarks,
            warped_moving_landmarks,
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
                normalize="image" in arr_name,  # label's value is already in [0, 1]
            )
os.chdir(MAIN_PATH)
