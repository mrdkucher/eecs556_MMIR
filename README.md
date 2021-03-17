# EECS 556 Final Project:
### Multimodal Image Registration: Comparison of Methods for 3D MRI to 3D Ultrasound Image Registration with Classical and Deep-Learning Accelerated Approaches

Dinank Gupta, David Kucher, Daniel Manwiller, Ellen Yeats

## LC2
### Running LC2 Code:
1) First install deepreg locally. Ensure you're in dir 'DeepReg' and run:
    ```bash
    cd DeepReg
    pip install -e . --no-cache-dir
    ```
2) Install Py-BOBYQA
    ```bash
    pip install Py-BOBYQA
    ```
3) Prepare dataset with landmarks
    ```bash
    c3d ../RESECT/NIFTI/Case1/US/Case1-US-before.nii.gz ../RESECT/NIFTI/Case1/MRI/Case1-T1.nii.gz -reslice-identity -resample-mm 0.5x0.5x0.5mm -o demos/lc2_paired_mrus_brain/Case1-MRI_in_US-rs.nii.gz

    c3d ../RESECT/NIFTI/Case1/US/Case1-US-before.nii.gz -resample-mm 0.5x0.5x0.5mm -o demos/lc2_paired_mrus_brain/Case1-US-rs.nii.gz

    python ../landmarks_split_txt.py --inputtag ../RESECT/NIFTI/Case1/landmarks/Case1-MRI-beforeUS.tag --savetxt demos/lc2_paired_mrus_brain/Case1_lm

    cd demos/lc2_paired_mrus_brain/

    c3d Case1-MRI_in_US-rs.nii.gz -scale 0 -landmarks-to-spheres Case1_lm_mri.txt 1-o Case1-MRI-landmarks-rs.nii.gz

    c3d Case1-US-rs.nii.gz -scale 0 -landmarks-to-spheres Case1_lm_us.txt 1-o Case1-US-landmarks-rs.nii.gz

    cd ../..
    ```
4) Run LC2 with:
    ```bash
    python demos/lc2_paired_mrus_brain/register.py -f Case1-US-rs.nii.gz -m Case1-MRI_in_US-rs.nii.gz -lf Case1-US-landmarks-rs.nii.gz -lm Case1-MRI-landmarks-rs.nii.gz -s 70 70 70 --verbose_bobyqa --max_iter 2000
    ```
5) Extract Landmarks coords from warped moving landmarks (not working)
    ```bash
    python ../landmarks_centre_mass.py --inputnii demos/lc2_paired_mrus_brain/logs_reg/moving_landmarks.nii.gz --movingnii demos/lc2_paired_mrus_brain/logs_reg/warped_moving_landmarks.nii.gz --savetxt demos/lc2_paired_mrus_brain/logs_reg/Case1-results
    ```

### Debugging LC2
1) Run LC2 on phantom images (extruded in 3d)
    ```bash
    cd DeepReg/demos/lc2_paired_mrus_brain
    python register.py -f phantom.nii.gz -m phantom_rot.nii.gz --verbose_bobyqa --max_iter 10000 -s 64 64 21 -g
    ```
2) Output is in logs_reg

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
