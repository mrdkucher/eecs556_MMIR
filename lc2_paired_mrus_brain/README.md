# Classical affine registration via LC2 for MRI and US images

This is a walkthrough of how to use our LC2[1] approach uses the DeepReg[2] framework and a custom similarity metric + optimization algorithm to perform classical affine image registration on a pair of multimodal images. The US image is fixed, while the MRI is deformed (moving image) to optimal alignment.

## Instructions
### Running LC2 Code on individual cases:
1) Make sure DeepReg is installed locally:
    ```bash
    cd DeepReg
    pip install -e . --no-cache-dir
    cd ..
    ```
2) Install Py-BOBYQA:
    ```bash
    pip install Py-BOBYQA
    ```
3) Ensure dataset is prepared as described in parent directory
4) Run LC2 with:
    ```bash
    python lc2_paired_mrus_brain/register.py -f RESECT/preprocessed/test/fixed_images/Case1.nii.gz -m RESECT/preprocessed/test/moving_images/Case1.nii.gz -lf RESECT/preprocessed/test/fixed_labels/Case1.nii.gz -lm RESECT/preprocessed/test/moving_labels/Case1.nii.gz -t RESECT/preprocessed/test/landmarks/Case1-MRI-breforeUS.tag -s 70 70 70 --verbose-bobyqa -g --max-iter 2000 -o case1_logs_reg
    ```
5) The output includes mTRE as text. Check lc2_paired_mrus_brain/case1_logs_reg for:
    - Fixed and moving images, labels, and warped moving images and labels.
    - mTRE, execution time, and BOBYQA output in reg_results.txt
    - PNG slices of each volume

### Debugging LC2
1) Run LC2 on phantom images (extruded in 3D)
    ```bash
    cd lc2_paired_mrus_brain
    python register.py -f phantom.nii.gz -m phantom_aff_noisy.nii.gz --verbose-bobyqa --max-iter 10000 -s 64 64 21 -g -o phantom_reg
    ```
2) Output is in logs_reg

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
[1] Fuerst et al., (2014). Automatic ultrasound–MRI registration for neurosurgery using the 2D and 3D LC2 Metric, Med. Im. Anl., Vol. 18, Is. 8, 1312-1319, ISSN 1361-8415, https://doi.org/10.1016/j.media.2014.04.008.


[2] Fu et al., (2020). DeepReg: a deep learning toolkit for medical image registration. Journal of Open Source Software, 5(55), 2705, https://doi.org/10.21105/joss.02705

[3] Xiao, Y., Fortin, M., Unsgård, G., Rivaz, H. and Reinertsen, I. (2017), REtroSpective Evaluation of Cerebral Tumors (RESECT): A clinical database of pre‐operative MRI and intra‐operative ultrasound in low‐grade glioma surgeries. Med. Phys., 44: 3875-3882. https://doi.org/10.1002/mp.12268