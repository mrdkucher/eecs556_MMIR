# Apply all preprocessing to images, then save to directories according to DeepReg train/val/test split
if [[ ! -e "RESECT/preprocessed" ]]; then
    mkdir RESECT/preprocessed
    mkdir RESECT/preprocessed/train
    mkdir RESECT/preprocessed/train/fixed_images
    mkdir RESECT/preprocessed/train/fixed_labels
    mkdir RESECT/preprocessed/train/moving_images
    mkdir RESECT/preprocessed/train/moving_labels
    mkdir RESECT/preprocessed/train/landmarks
    mkdir RESECT/preprocessed/valid
    mkdir RESECT/preprocessed/valid/fixed_images
    mkdir RESECT/preprocessed/valid/fixed_labels
    mkdir RESECT/preprocessed/valid/moving_images
    mkdir RESECT/preprocessed/valid/moving_labels
    mkdir RESECT/preprocessed/valid/landmarks
    mkdir RESECT/preprocessed/test
    mkdir RESECT/preprocessed/test/fixed_images
    mkdir RESECT/preprocessed/test/fixed_labels
    mkdir RESECT/preprocessed/test/moving_images
    mkdir RESECT/preprocessed/test/moving_labels
    mkdir RESECT/preprocessed/test/landmarks
else
    echo "preprocessed already exists"
fi

# Create train dataset
train_cases=(3 4 5 6 7 8 14 15 16 17 18 19 24 25 26 27)
for i in "${train_cases[@]}"; do
    echo "Resampling: RESECT/NIFTI/Case$i/US/Case$i-US-before.nii.gz"
    c3d RESECT/NIFTI/Case$i/US/Case$i-US-before.nii.gz RESECT/NIFTI/Case$i/MRI/Case$i-T1.nii.gz -reslice-identity -resample-mm 0.5x0.5x0.5mm -o RESECT/preprocessed/train/moving_images/Case$i.nii.gz
    flair_index=$(seq -f "%02g" $i $i)
    c3d RESECT/NIFTI/Case$i/US/Case$i-US-before.nii.gz RESECT/NIFTI/Case$i/MRI/Case$i-FLAIR.nii.gz -reslice-identity -resample-mm 0.5x0.5x0.5mm -o RESECT/preprocessed/train/moving_images/Case1$flair_index.nii.gz
    c3d RESECT/NIFTI/Case$i/US/Case$i-US-before.nii.gz -resample-mm 0.5x0.5x0.5mm -o RESECT/preprocessed/train/fixed_images/Case$i.nii.gz
    python landmarks_split_txt.py --inputtag RESECT/NIFTI/Case$i/Landmarks/Case$i-MRI-beforeUS.tag --savetxt lm
    c3d RESECT/preprocessed/train/moving_images/Case$i.nii.gz -scale 0 -landmarks-to-spheres lm_mri.txt 1.25 -o RESECT/preprocessed/train/moving_labels/Case$i.nii.gz
    c3d RESECT/preprocessed/train/fixed_images/Case$i.nii.gz -scale 0 -landmarks-to-spheres lm_us.txt 1.25 -o RESECT/preprocessed/train/fixed_labels/Case$i.nii.gz
    cp RESECT/NIFTI/Case$i/Landmarks/Case$i-MRI-beforeUS.tag  RESECT/preprocessed/train/landmarks/Case$i-MRI-beforeUS.tag
    # Cleanup
    rm lm_mri.txt lm_us.txt
done

# Create val dataset
val_cases=(1 12 13)
for i in "${val_cases[@]}"; do
    echo "Resampling: RESECT/NIFTI/Case$i/US/Case$i-US-before.nii.gz"
    c3d RESECT/NIFTI/Case$i/US/Case$i-US-before.nii.gz RESECT/NIFTI/Case$i/MRI/Case$i-T1.nii.gz -reslice-identity -resample-mm 0.5x0.5x0.5mm -o RESECT/preprocessed/valid/moving_images/Case$i.nii.gz
    flair_index=$(seq -f "%02g" $i $i)
    c3d RESECT/NIFTI/Case$i/US/Case$i-US-before.nii.gz RESECT/NIFTI/Case$i/MRI/Case$i-FLAIR.nii.gz -reslice-identity -resample-mm 0.5x0.5x0.5mm -o RESECT/preprocessed/valid/moving_images/Case1$flair_index.nii.gz
    c3d RESECT/NIFTI/Case$i/US/Case$i-US-before.nii.gz -resample-mm 0.5x0.5x0.5mm -o RESECT/preprocessed/valid/fixed_images/Case$i.nii.gz
    python landmarks_split_txt.py --inputtag RESECT/NIFTI/Case$i/Landmarks/Case$i-MRI-beforeUS.tag --savetxt lm
    c3d RESECT/preprocessed/valid/moving_images/Case$i.nii.gz -scale 0 -landmarks-to-spheres lm_mri.txt 1.25 -o RESECT/preprocessed/valid/moving_labels/Case$i.nii.gz
    c3d RESECT/preprocessed/valid/fixed_images/Case$i.nii.gz -scale 0 -landmarks-to-spheres lm_us.txt 1.25 -o RESECT/preprocessed/valid/fixed_labels/Case$i.nii.gz
    cp RESECT/NIFTI/Case$i/Landmarks/Case$i-MRI-beforeUS.tag  RESECT/preprocessed/valid/landmarks/
    # Cleanup
    rm lm_mri.txt lm_us.txt
done

# Create test dataset
test_cases=(2 21 23)
for i in "${test_cases[@]}"; do
    echo "Resampling: RESECT/NIFTI/Case$i/US/Case$i-US-before.nii.gz"
    c3d RESECT/NIFTI/Case$i/US/Case$i-US-before.nii.gz RESECT/NIFTI/Case$i/MRI/Case$i-T1.nii.gz -reslice-identity -resample-mm 0.5x0.5x0.5mm -o RESECT/preprocessed/test/moving_images/Case$i.nii.gz
    flair_index=$(seq -f "%02g" $i $i)
    c3d RESECT/NIFTI/Case$i/US/Case$i-US-before.nii.gz RESECT/NIFTI/Case$i/MRI/Case$i-FLAIR.nii.gz -reslice-identity -resample-mm 0.5x0.5x0.5mm -o RESECT/preprocessed/test/moving_images/Case1$flair_index.nii.gz
    c3d RESECT/NIFTI/Case$i/US/Case$i-US-before.nii.gz -resample-mm 0.5x0.5x0.5mm -o RESECT/preprocessed/test/fixed_images/Case$i.nii.gz
    python landmarks_split_txt.py --inputtag RESECT/NIFTI/Case$i/Landmarks/Case$i-MRI-beforeUS.tag --savetxt lm
    c3d RESECT/preprocessed/test/moving_images/Case$i.nii.gz -scale 0 -landmarks-to-spheres lm_mri.txt 1.25 -o RESECT/preprocessed/test/moving_labels/Case$i.nii.gz
    c3d RESECT/preprocessed/test/fixed_images/Case$i.nii.gz -scale 0 -landmarks-to-spheres lm_us.txt 1.25 -o RESECT/preprocessed/test/fixed_labels/Case$i.nii.gz
    cp RESECT/NIFTI/Case$i/Landmarks/Case$i-MRI-beforeUS.tag  RESECT/preprocessed/test/landmarks/
    # Cleanup
    rm lm_mri.txt lm_us.txt
done
