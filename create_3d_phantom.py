import nibabel as nib
import numpy as np
from skimage.data import shepp_logan_phantom
import tensorflow as tf
import deepreg.model.layer_util as layer_util


# get downsampled phantom
p = shepp_logan_phantom()[::4, ::4, np.newaxis]
p3d = np.repeat(p, 42, axis=2)
# zero pad the image (10 voxels on each axis)
p3d = np.pad(p3d, (10, 10))
img = nib.Nifti1Image(p3d, affine=np.eye(4))
nib.save(img, 'phantom.nii.gz')

# Create artificial landmarks
N = 15
landmarks = np.array(
    [np.random.uniform(high=99, size=N),     # x: 0-99
     np.random.uniform(high=99, size=N),     # y: 0-99
     np.random.uniform(high=41, size=N)]).T  # z: 0-41
print(landmarks.shape)
homogeneous_landmarks = np.concatenate((landmarks, np.ones((N, 1))), axis=1)

# perform affine transformation on one image
affine = np.array(
    [[1.0, 0.0, 0.0, 1.0],
     [0.0, 1.1, 0.1, 1.0],
     [0.0, 0.1, 1.1, 1.0],
     [0.0, 0.0, 0.0, 1.0]]
)
# get transformed landmarks
warped_homogeneous_landmarks = homogeneous_landmarks @ affine.T
warped_landmarks = warped_homogeneous_landmarks[:, :3]

# Need transpose of first 3 rows to warp the reference grid
tf_aff = tf.convert_to_tensor(affine[:3, :].T[np.newaxis, :, :], dtype=tf.float32)
grid_ref = layer_util.get_reference_grid(grid_size=p3d.shape)
grid_warp = layer_util.warp_grid(grid_ref, tf_aff)

# Add gaussian noise to the image (only to image data, not to zeros)
mask = p3d > 0
p3d_noisy = p3d + 0.1 * np.random.normal(0, 1, size=p3d.shape)
p3d_noisy *= mask
p3d_noisy = tf.convert_to_tensor(p3d_noisy[np.newaxis, :, :, :], dtype=tf.float32)

p3d2 = layer_util.resample(p3d_noisy, grid_warp)
p3d2 = p3d2.numpy().squeeze()
img2 = nib.Nifti1Image(p3d2, affine=np.eye(4))
nib.save(img2, 'phantom_aff_noisy.nii.gz')

with open("phantom.tag", "w") as f:
    f.writelines(["Tag Point File\n", "Volumes = 2;\n", "% Phantom\n", "\n", "Points = \n"])
    for i in range(N):
        for j in range(3):
            f.write("{:>11.6f}".format(landmarks[i, j]))
        for j in range(3):
            f.write("{:>11.6f}".format(warped_landmarks[i, j]))
        f.write("\n")
