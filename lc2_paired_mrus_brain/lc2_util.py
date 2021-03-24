import nibabel as nib
import numpy as np
import tensorflow as tf

from deepreg.registry import REGISTRY
import deepreg.model.layer_util as layer_util


def load_preprocess_image(file_path, image_size=[-1, -1, -1], normalize=True, fixed=False, method='bilinear'):
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
    if image_size != [-1, -1, -1]:
        image = layer_util.resize3d(image, image_size, method=method)
        scale = orig_shape / np.array(image_size)
        # convert to homogeneous coords
        scale = np.concatenate((scale, np.ones(1)))
        # apply scaling to image affine
        image_aff = image_aff @ np.diag(scale)

    if normalize:
        # normalize to [0, 1]
        image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
        if fixed:
            image = image + 1e-10  # add constant so it's not all 0s

    return image, image_aff


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


# BOBYQA optimization:
def build_objective_function(grid, mov, fix, image_loss_config={"name": "lc2"}) -> object:
    """
    Builder function for the function which BOBYQA optimizes

    :param grid: reference grid return from layer_util.get_reference_grid
    :param mov: moving image [1, m_dim1, m_dim2, m_dim3]
    :param fix: fixed image [1, f_dim1, f_dim2, f_dim3]
    :param image_loss_config: dictionary of config {"name": "lc2"}
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
