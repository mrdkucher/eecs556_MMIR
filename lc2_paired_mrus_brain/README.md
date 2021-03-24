# Classical affine registration via LC2 for MRI and US paired images


TODO: update this for LC2 and BOBYQA

This is a demo that uses the DeepReg[1] package for classical affine image
registration, which iteratively solves an optimisation problem. Gradient descent is used
to minimise the image dissimilarity function of a given pair of moving anf fixed images.

## Author

David Kucher

## Application

Aligning pre-operative MRI to intra-operative US images.

## Data

Data is from Xiao et al.[2]: [EASY-RESECT](https://archive.sigma2.no/pages/public/datasetDetail.jsf?id=10.11582/2020.00025).

## Instruction
From `DeepReg/` directory, download and preprocess data, then register the images with:

```bash
python demos/lc2_paired_mrus_brain/demo_data.py
python demos/lc2_paired_mrus_brain/demo_register.py
```


## Visualise

The following command can be executed to generate a plot of three image slices from the
the moving image, warped image and fixed image (left to right) to visualise the
registration. Please see the visualisation tool docs
[here](https://github.com/DeepRegNet/DeepReg/blob/main/docs/source/docs/visualisation_tool.md)
for more visualisation options such as animated gifs.

TODO: visualization command

[comment]: # (```bash)
[comment]: # (deepreg_vi -m 2 -i 'demos/lc2_paired_mrus_brain/logs_reg/Case1.nii.gz, demos/classical_ct_headneck_affine/logs_reg/warped_moving_image.nii.gz, demos/classical_ct_headneck_affine/logs_reg/fixed_image.nii.gz' --slice-inds '4,8,12' -s demos/classical_ct_headneck_affine/logs_reg)
[comment]: # (```)

Note: The registration script must be run before running the command to generate the
visualisation.

![plot](../assets/classical_ct_headneck_affine.png)


## Reference
[1] Fu et al., (2020). DeepReg: a deep learning toolkit for medical image registration. Journal of Open Source Software, 5(55), 2705, https://doi.org/10.21105/joss.02705

[2] Xiao, Y., Fortin, M., Unsgård, G., Rivaz, H., Reinertsen, I. (2020).EASY-RESECT [Data set]. Norstore. https://doi.org/10.11582/2020.00025
