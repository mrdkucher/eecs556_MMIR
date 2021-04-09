# EECS 556 Final Project:
### Multimodal Image Registration: Comparison of Methods for 3D MRI to 3D Ultrasound Image Registration with Classical and Deep-Learning Accelerated Approaches

Dinank Gupta, David Kucher, Daniel Manwiller, Ellen Yeats

## Data Preparation
### Automated:
1) Download [RESECT Dataset](https://archive.norstore.no/pages/public/datasetDetail.jsf?id=10.11582/2017.00004) and place next to preprocessRESECT.sh.
2) Run preprocessing via ```./preprocessRESECT.sh``` to:
    - Resample US to 0.5mmx0.5mmx0.5mm
    - Reslice, resample, and transform MRI to be in same coordinates as US
    - Read landmark files, and create voxelized landmarks for US and MRI images with value == landmark ID
3) Data is split into train, validation, and test sets, with fixed/moving images, labels (voxelized landmarks), and landmarks (pts).

### Manually, taking case 1 as an example:
1) Reslice and resample MRI to 0.5x0.5x0.5mm and to US coordinates
```
bash c3d RESECT/NIFTI/Case1/US/Case1-US-before.nii.gz RESECT/NIFTI/Case1/MRI/Case1-T1.nii.gz -reslice-identity -resample-mm 0.5x0.5x0.5mm -o Case1-MRI_in_US-rs.nii.gz
```

2) Resample US to 0.5x0.5x0.5mm
```bash
c3d RESECT/NIFTI/Case1/US/Case1-US-before.nii.gz -resample-mm 0.5x0.5x0.5mm -o Case1-US-rs.nii.gz
```

3) Extract landmarks to txt files
```bash
python landmarks_split_txt.py --inputtag RESECT/NIFTI/Case1/Landmarks/Case1-MRI-beforeUS.tag --savetxt Case1_lm
```

4) Create voxelized landmarks as labels for MRI
```bash 
c3d Case1-MRI_in_US-rs.nii.gz -scale 0 -landmarks-to-spheres Case1_lm_mri.txt 2 -o Case1-MRI-landmarks-rs.nii.gz
```

5) Create voxelized landmarks as labels for US
```bash
c3d Case1-US-rs.nii.gz -scale 0 -landmarks-to-spheres Case1_lm_us.txt 2 -o Case1-US-landmarks-rs.nii.gz
```

## LC2
### Running LC2 Code:
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
3) Ensure Dataset is prepared as described above
4) Run LC2 with:
    ```bash
    python lc2_paired_mrus_brain/register.py -f RESECT/preprocessed/test/fixed_images/Case1.nii.gz -m RESECT/preprocessed/test/moving_images/Case1.nii.gz -lf RESECT/preprocessed/test/fixed_labels/Case1.nii.gz -lm RESECT/preprocessed/test/moving_labels/Case1.nii.gz -t RESECT/preprocessed/test/landmarks/Case1-MRI-breforeUS.tag -s 70 70 70 --verbose-bobyqa -g --max-iter 2000 -o case1_logs_reg
    ```
5) The output includes mTRE as text. Check lc2_paired_mrus_brain/logs_reg for:
    - Fixed and moving images, labels, and warped moving images and labels.
    - The affine transformation applied to the moving image to warp it
    - PNG slices of each volume


## c3d utility
### Python script for quickly separating the .tag file into a .txt

    ```bash
    python landmarks_split_txt.py --inputtag *folder*/Case1-MRI-beforeUS.tag --savetxt Case1_lm

    ```
To get the MRI resliced into the US img coordinate system, run:
    ```bash
c3d Case1-US-before.nii.gz Case1-FLAIR.nii.gz
    -reslice-identity -resample-mm 0.5x0.5x0.5mm -o Case1-MRI_in_US.nii.gz
    ```
To get the US resliced to the finer 0.5 mm resolution, run:
   ```bash
c3d Case1-US-before.nii.gz -resample-mm 0.5x0.5x0.5mm -o Case1-US.nii.gz
   ```
    If it's helpful for your framework, you can then run:
    ```bash
    c3d Case1-MRI_in_US.nii.gz -scale 0 -landmarks-to-spheres Case1_lm_mri.txt 1-o Case1-MRI-landmarks.nii.gz
    ```
Running c3d with that command will create a new .nii.gz with voxel spheres representing the landmarks. You can then apply your transformation to that file directly.
    
### Python script for finding the coordinates of the spheres from the COM

   ```bash
   python landmarks_centre_mass.py --inputnii Case1-MRI-landmarks.nii.gz --movingnii Case1-MRI-deformed_landmarks.nii.gz --savetxt Case1-results
   ```

### Git Steps:
Setup:
- `git clone https://github.com/mrdkucher/eecs556_MMIR.git`

Making changes:
- Add all changes to be committed: `git add .` or `git add <filename 1> <filename 2> ... <filename N>`
- Commit local changes to your local repo with message: `git commit -m "<commit message>"`
- Rebase any changes from remote: `git pull --rebase origin master`
  - If there are merge conflicts, resolve them by keeping whatever code should stay in
  - continue rebase by running: `git add .` and `git rebase --continue`
  - at end of rebase, you'll be prompted to update the commit message, which you can leave alone.
- Push local changes to remote branch: `git push -u origin master`, or just `git push` after you've done the former command once.

In summary:
- Make changes
- `git add .`
- `git commit -m "<message>"`
- `git pull --rebase origin master`
- `git push`
