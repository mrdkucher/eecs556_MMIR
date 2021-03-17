import nibabel as nib
import numpy as np
from skimage.data import shepp_logan_phantom


# get downsampled phantom
p = shepp_logan_phantom()[::4, ::4, np.newaxis]
p3d = np.repeat(p, 42, axis=2)
img = nib.Nifti1Image(p3d, affine=np.eye(4))
nib.save(img, 'phantom.nii.gz')

# perform rotation on one image
p3d2 = np.rot90(p3d, axes=(0, 1))
img2 = nib.Nifti1Image(p3d2, affine=np.eye(4))
nib.save(img2, 'phantom_rot.nii.gz')
