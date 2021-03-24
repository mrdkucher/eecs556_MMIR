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

import deepreg.model.layer_util as layer_util
import deepreg.util as util
from lc2_util import load_preprocess_image, build_objective_function, calculate_mTRE, extract_centroid

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

# load and preprocess fixed and moving images
moving_image, moving_image_aff = load_preprocess_image(MOVING_PATH, image_size=args.image_size)
fixed_image, fixed_image_aff = load_preprocess_image(FIXED_PATH, image_size=args.image_size, fixed=True)

if use_landmarks:
    # load and prepreprocess fixed and moving landmarks (images)
    # resize wth 'nearest' interp: need integer values 1-15
    moving_lm_img, _ = load_preprocess_image(MOVING_LM_PATH, image_size=args.image_size,
                                             normalize=False, method='nearest')
    fixed_lm_img, _ = load_preprocess_image(FIXED_LM_PATH, image_size=args.image_size,
                                            normalize=False, method='nearest')
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
print("seek global minimum:", args.seek_global_minimum)
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