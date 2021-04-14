import nibabel as nib
import numpy as np
import tensorflow as tf
# from tensorflow import tfg
import scipy.interpolate

from deepreg.registry import REGISTRY
import deepreg.model.layer_util as layer_util
import deepreg.util as util


def load_preprocess_image(file_path, image_size=[-1, -1, -1], normalize=True, fixed=False, method='bilinear'):
    '''
    Load in nifti image, save as a tensor, resize image
    to image_size given in args, and (optionally) normalize.

    :param file_path: path to .nii.gz file (str)
    :param normalize: normalize image to [0, 1] (bool)
    :param method: image resize method. See tf.image.ResizeMethod (str)
    :return: image tensor (1, h, w, d)
    '''
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


def warp_landmarks(moving_landmarks, moving_affine, reg_affine, fixed_affine):
    '''
    Warp world coordinate landmarks according to reg_affine.

    :param moving_landmarks: (numpy array) moving landmarks
        - shape: (n, 3)
    :param moving_affine: (numpy array) moving affine matrix (voxel coords -> world coords)
        - shape: (4, 4)
    :param reg_affine: (numpy array) registration affine matrix (voxel coords -> voxel coords))
        - shape: (4, 3)
    :param fixed_affine: (numpy array) fixed affine matrix (voxel coords -> world coords)
        - shape: (4, 4)

    :return warped_landmarks: numpy array of warped moving landmarks
        - shape: (n, 3)
    '''
    num_lms = moving_landmarks.shape[0]
    bias = np.ones((num_lms, 1))
    # Landmark - based:
    # Landmarks -> [inverse moving_image affine] -> landmarks in voxels -> [affine xform] ->
    # warped landmarks in voxels -> [fixed_image affine] -> warped moving landmarks
    # convert to homogeneous coordinates
    mov_hlms = np.concatenate((moving_landmarks, bias), axis=1)
    # convert to pixel values from world coords
    mov_hvox = mov_hlms @ np.linalg.inv(moving_affine).T
    # perform transformation
    warped_moving_voxels = mov_hvox @ reg_affine
    # transform back to world coords
    warped_moving_landmarks = np.concatenate(
        (warped_moving_voxels, bias), axis=1) @ fixed_affine.T
    warped_moving_landmarks = warped_moving_landmarks[:, :3]
    return warped_moving_landmarks


def extract_centroid(image):
    '''
    Extract centroid from nifti images with landmark spheres
    which have integer values according to labels
    Adapted from: https://gist.github.com/mattiaspaul/f4183f525b1cbc65e71ad23298d6436e

    :param image:
        - shape: (dim_1, dim_2, dim_3) or (batch, dim_1, dim_2, dim_3)
        - tensor or numpy array

    :return positions:
        - numpy array of labels 1
    '''
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


def warp_image(image, grid_ref, var_affine, method='linear'):
    '''
    Warp an image according to var affine. Warps the reference
    grid with var_affine, then resamples the image. Uses 'method'
    for interpolation

    :param image: (tf tensor) image
        - shape: (x, y, z)
    :param grid_ref: (tf tensor) reference grid (mesh grid)
        - shape: (x, y, z, 3)
    :param var_affine: (tf tensor) affine transformation
        - shape: (4, 3)
    :param method: (str) interpolation method. 'nearest', 'linear', or 'cubic'

    :return warped_image = (tf tensor) warped image
        - shape: (x, y, z)
    '''
    # get sample points by using the affine transformation on a reference grid
    sample_pts = layer_util.warp_grid(grid_ref, var_affine)
    if method == 'linear':
        # Input; [A1, ..., An, H, W, D, C]
        # sample_pts = tf.reshape(sample_pts, [1, *sample_pts.shape])
        # warped_image = tfg.math.interpolation.trilinear.interpolate(image, sample_pts)
        warped_image = layer_util.resample(vol=image, loc=sample_pts)
    elif method == 'cubic':  # TODO
        pass
    elif method == 'nearest':  # Use for labels
        # use nearest neighbor interpolation so landmarks can be properly extracted later
        grid_ref_np = grid_ref.numpy().reshape((-1, 3))
        image_np = image.numpy().squeeze().reshape(-1)
        grid_opt_np = sample_pts.numpy().squeeze().reshape((-1, 3))
        warped_image = scipy.interpolate.griddata(
            grid_ref_np, image_np, grid_opt_np, method='nearest')
        warped_image = warped_image.reshape(image.shape)
        warped_image = tf.convert_to_tensor(warped_image, dtype=tf.float32)
    else:
        raise NotImplementedError('Method: \'' + str(method) + '\' not implemented for interpolation')
    return warped_image


def labels2world_coords(moving_labels, warped_moving_labels, fixed_labels, fixed_affine):
    '''
    Warp world coordinate landmarks according to reg_affine.

    :param labels: (numpy array) moving labels (spheres)
        - shape: (x, y, z)
    :param warped_moving_labels: (numpy array) warped moving labels (speheres)
        - shape: (x, y, z)
    :param fixed_labels: (numpy array) fixed labels (spheres)
        - shape: (x, y, z)
    :param fixed_affine: (numpy array) fixed affine matrix (voxel coords -> world coords)
        - shape: (4, 4)

    :return world_coords = (moving, warped_moving, fixed): (tuple) of 3 numpy arrays
        - moving, warped_moving, fixed shape: (n, 3)
    '''
    world_coords = []
    for labels in [moving_labels, warped_moving_labels, fixed_labels]:
        # Sphere - based:
        # Extract warped centroids
        label_centroid_voxels = extract_centroid(
            labels)
        num_lms = label_centroid_voxels.shape[0]
        bias = np.ones((num_lms, 1))
        # Convert to world coords
        label_centroid_world = np.concatenate(
            (label_centroid_voxels, bias), axis=1) @ fixed_affine.T
        label_centroid_world = label_centroid_world[:, :3]
        # Save world coords
        world_coords.append(label_centroid_world)
    return tuple(world_coords)


def calculate_mTRE(xyz_true, xyz_predict):
    '''
    Calculate mTRE (mean Target Registration Error) between
    two sets of points.

    :param xyz_true: (numpy array) fixed landmarks
        - shape: (n, 3) or (n, 4) for homogeneous coords
    :param xyz_predict: (numpy array) warped moving landmarks
        - shape: (n, 3) or (n, 4) for homogeneous coords
    :return mTRE: (float) mean euclidean distance between two sets of points
    '''
    assert xyz_true.shape == xyz_predict.shape
    TRE = np.sqrt(np.sum(np.power(xyz_true - xyz_predict, 2), axis=1))
    mTRE = np.mean(TRE)
    return mTRE


# Create transformation matrix:
def create_transformation_mat(weights, transformation_type='affine'):
    if transformation_type not in ['affine', 'rigid', 'translate']:
        raise NotImplementedError('Transformation: ' + str(transformation_type) + ' not available.')

    transform = np.eye(4, 3)
    if transformation_type == 'translate':
        transform[3, :] = weights
    if transformation_type == 'rigid':
        rx, ry, rz = weights[:3]

        Rx = np.array([[1,          0,           0],  # noqa: E241
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx),  np.cos(rx)]])  # noqa: E201, E241
        Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],  # noqa: E201, E241
                       [          0, 1,          0],  # noqa: E201, E241
                       [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz),  np.cos(rz), 0],  # noqa: E241
                       [          0,          0, 1]])  # noqa: E201, E241
        transform[:3, :] = Rx @ Ry @ Rz
        transform[3, :] = weights[3:]
    if transformation_type == 'affine':
        transform = weights.reshape(4, 3)
    return transform


def save_image(image, name, SAVE_PATH):
    '''
    Save image (volume) as slices to PNG files, and save image as nifti

    :param image: (tf tensor) image to save
        - shape: (x, y, z)
    :param image: (str) file name
    :param SAVE_PATH: (str) path to output directory
    '''
    arr = tf.transpose(image, [1, 2, 3, 0]) if image.ndim == 4 else tf.squeeze(image)
    for n in range(arr.shape[-1]):
        util.save_array(
            save_dir=SAVE_PATH,
            arr=arr[..., n],
            name=name + (arr.shape[-1] > 1) * '_{}'.format(n),
            # label's value is already in [0, 1]
            normalize='image' in name,
        )


# BOBYQA optimization:
def build_objective_function(grid, mov, fix, image_loss_config={'name': 'lc2'}, transformation_type='affine') -> object:
    '''
    Builder function for the function which BOBYQA optimizes

    :param grid: reference grid return from layer_util.get_reference_grid
    :param mov: moving image [1, m_dim1, m_dim2, m_dim3]
    :param fix: fixed image [1, f_dim1, f_dim2, f_dim3]
    :param image_loss_config: dictionary of config for loss function {'name': 'lc2'}
    :param transformation_type: str. One of ['affine', 'rigid', 'translate'].
    :return loss: image dissimilarity
    '''
    loss_fn = REGISTRY.build_loss(config=image_loss_config)

    if transformation_type not in ['affine', 'rigid', 'translate']:
        raise NotImplementedError('Transformation: ' + str(transformation_type) + ' not available.')

    def objective_function(weights) -> object:
        '''
        Objective function for BOBYQA to minimize:
        Performs affine transformation on mov, defined by weights:

        :param weights: 1D numpy array
            transformation types:
                translate: [t1, t2, t3]
                rigid: [r1, r2, r3, t1, t2, t3]
                affine: [a11, a21, a31,
                         a12, a22, a32],
                         a13, a23, a33],
                          t1,  t2,  t3]

            convert to shape [1, 4, 3],
            transpose of normal affine matrix:
            y_pred = [y_true | 1] * weights
            weights = [[a11, a21, a31],
                       [a12, a22, a32],
                       [a13, a23, a33],
                       [ t1,  t2,  t3]]
            where t1, t2, and t3 correspond to the bias weights
            (translation) for x, y, and z, accordingly

        :return: image dissimilarity measure, to minimize

        '''
        transform = create_transformation_mat(weights, transformation_type=transformation_type)
        transform = tf.convert_to_tensor(transform.reshape((1, 4, 3)), dtype=tf.float32)

        pred = layer_util.resample(
            vol=mov, loc=layer_util.warp_grid(grid, transform))
        return loss_fn(y_true=fix, y_pred=pred)

    return objective_function
