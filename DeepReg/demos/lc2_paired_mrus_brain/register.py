"""
A DeepReg Demo for classical affine iterative pairwise registration algorithms
"""
import argparse
import os
import shutil

# import h5py
import tensorflow as tf

import deepreg.model.layer_util as layer_util
import deepreg.util as util
from deepreg.registry import REGISTRY
from deepreg.dataset.loader.nifti_loader import load_nifti_file

import pybobyqa
import numpy as np

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
    "-s", "--image_size",
    help="3-entry tuple to resize image e.g. (256, 256, 288)",
    dest="image_size",
    nargs=3,
    action="store",
    type=int,
    default=[64, 64, 72]
)
parser.add_argument(
    "-n", "--n_iter",
    help="number of iterations to run",
    dest="n_iter",
    action="store",
    type=int,
    default=1000
)
args = parser.parse_args()

MAIN_PATH = os.getcwd()
PROJECT_DIR = "demos/lc2_paired_mrus_brain"
os.chdir(PROJECT_DIR)

DATA_PATH = "dataset"
MOVING_PATH = os.path.join(DATA_PATH, args.moving)
FIXED_PATH = os.path.join(DATA_PATH, args.fixed)

# registration parameters
image_loss_config = {"name": "lc2"}
learning_rate = 1e-3
total_iter = args.n_iter

# load image
if not os.path.exists(DATA_PATH):
    raise ("Download the data using demo_data.py script")
if not os.path.exists(MOVING_PATH):
    raise FileNotFoundError(f"Moving image not fount at: {MOVING_PATH}")
if not os.path.exists(FIXED_PATH):
    raise FileNotFoundError(f"Fixed image not fount at: {FIXED_PATH}")


def load_preprocess_image(file_path, normalize=True):
    image = load_nifti_file(file_path)
    image = tf.expand_dims(tf.convert_to_tensor(image), axis=0)

    # resize to arg.image_size
    image = layer_util.resize3d(image, args.image_size)

    if normalize:
        # normalize to [0, 1]
        image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))

    return image


# load and preprocess fixed and moving images: DAK set normalize to false for now...
moving_image = load_preprocess_image(MOVING_PATH)
fixed_image = load_preprocess_image(FIXED_PATH)

# Get a reference grid for warping
fixed_image_size = fixed_image.shape
grid_ref = layer_util.get_reference_grid(grid_size=fixed_image_size[1:4])


# # optimisation
# # @tf.function  # this turns execution into a graph, which isn't ok for LC2....
# def train_step(grid, weights, optimizer, mov, fix) -> object:
#     """
#     Train step function for backprop using gradient tape

#     :param grid: reference grid return from layer_util.get_reference_grid
#     :param weights: trainable affine parameters [1, 4, 3]
#     :param optimizer: tf.optimizers
#     :param mov: moving image [1, m_dim1, m_dim2, m_dim3]
#     :param fix: fixed image [1, f_dim1, f_dim2, f_dim3]
#     :return loss: image dissimilarity to minimise
#     """
#     with tf.GradientTape() as tape:
#         pred = layer_util.resample(vol=mov, loc=layer_util.warp_grid(grid, weights))
#         loss = REGISTRY.build_loss(config=image_loss_config)(
#             y_true=fix,
#             y_pred=pred,
#         )
#     gradients = tape.gradient(loss, [weights])  # DAK left off here -> zero division error during gradient descent. Time for BOBYQA!
#     optimizer.apply_gradients(zip(gradients, [weights]))
#     return loss

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
var_affine = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

obj_fn = build_objective_function(grid_ref, moving_image, fixed_image)
soln = pybobyqa.solve(obj_fn, var_affine, print_progress=True)
print(soln)

var_affine = tf.convert_to_tensor(soln.x.reshape((1, 4, 3)), dtype=tf.float32)

# warp the moving image using the optimised affine transformation
grid_opt = layer_util.warp_grid(grid_ref, var_affine)
warped_moving_image = layer_util.resample(vol=moving_image, loc=grid_opt)

# DAK TODO: get labels (landmarks) loaded in... I'll need to adjust them based on resizing the image,
# Then calculate TRE accordingly...

# # warp the moving labels using the optimised affine transformation
# warped_moving_labels = tf.stack(
#     [
#         layer_util.resample(vol=moving_labels[..., idx], loc=grid_opt)
#         for idx in range(fixed_labels.shape[4])
#     ],
#     axis=4,
# )

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
        # moving_labels,
        # fixed_labels,
        # warped_moving_labels,
    ]
]
arr_names = [
    "moving_image",
    "fixed_image",
    "warped_moving_image",
    # "moving_label",
    # "fixed_label",
    # "warped_moving_label",
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
