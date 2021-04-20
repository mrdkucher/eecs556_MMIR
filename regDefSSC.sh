#!/bin/bash

mmirpath="/home/yeatsem/Proj556/eecs556_MMIR" #put the path to the local version of the github repo here
c3dpath="/home/yeatsem/itk/bin" #put the path to c3d binary folder here
resectpath="/home/yeatsem/Proj556/RESECT" #put the path to RESECT data here

outfile="reg_results_deform_SSC.txt"

g_aff="6x5x4x3" #grid spacing
l_aff="4" #number of levels
L_aff="7x6x5x4"
q_aff="4x3x2x1"

l_def="4" #number of levels
g_def="6x5x4x3" #grid spacing
L_def="7x6x5x4" #maximum search radius
q_def="4x3x2x1" #quantization
a="1.6"

echo "affine params"
echo ${l_aff} >> ${outfile}
echo ${g_aff} >> ${outfile}
echo ${L_aff} >> ${outfile}
echo ${q_aff} >> ${outfile}

echo "deformable params"
echo ${l_def} >> ${outfile}
echo ${g_def} >> ${outfile}
echo ${L_def} >> ${outfile}
echo ${q_def} >> ${outfile}
echo ${a} >> ${outfile}

for c in {1..8} {12..19} 21 {23..27}
do
  #preprocess the RESECT data
  cd ${c3dpath}
  ./c3d  ${resectpath}/NIFTI/Case${c}/US/Case${c}-US-before.nii.gz  ${resectpath}/NIFTI/Case${c}/MRI/Case${c}-FLAIR.nii.gz -reslice-identity -resample-mm 0.5x0.5x0.5mm -o  ${resectpath}/NIFTI/Case${c}/Case${c}-MRI_in_US.nii.gz
  ./c3d  ${resectpath}/NIFTI/Case${c}/US/Case${c}-US-before.nii.gz -resample-mm 0.5x0.5x0.5mm -o  ${resectpath}/NIFTI/Case${c}/Case${c}-US.nii.gz

  #save landmark coordinates as .txt
  cd ${mmirpath}
  python landmarks_split_txt_SSC.py --inputtag ${resectpath}/NIFTI/Case${c}/Landmarks/Case${c}-MRI-beforeUS.tag --savetxt ${resectpath}/NIFTI/Case${c}/Case${c}_lm

  #generate MRI voxelized landmarks
  cd ${c3dpath}
  ./c3d ${resectpath}/NIFTI/Case${c}/Case${c}-MRI_in_US.nii.gz -scale 0 -landmarks-to-spheres ${resectpath}/NIFTI/Case${c}/Case${c}_lm_mri.txt 1 -o ${resectpath}/NIFTI/Case${c}/Case${c}-MRI-landmarks.nii.gz

  # generate US voxelized landmarks
  ./c3d ${resectpath}/NIFTI/Case${c}/Case${c}-US.nii.gz -scale 0 -landmarks-to-spheres ${resectpath}/NIFTI/Case${c}/Case${c}_lm_us.txt 1 -o ${resectpath}/NIFTI/Case${c}/Case${c}-US-landmarks.nii.gz

  echo "Calculating initial TRE"
  cd ${mmirpath}
  iniTRE=$(python landmarks_centre_mass_SSC.py --inputnii ${resectpath}/NIFTI/Case${c}/Case${c}-US-landmarks.nii.gz --movingnii ${resectpath}/NIFTI/Case${c}/Case${c}-MRI-landmarks.nii.gz --savetxt ${resectpath}/NIFTI/Case${c}/Case${c}-prelim)
  
  # run deformable deeds
  cd deeds
  defTime=$(./deedsBCV -F ${resectpath}/NIFTI/Case${c}/Case${c}-US.nii.gz -M ${resectpath}/NIFTI/Case${c}/Case${c}-MRI_in_US.nii.gz  -G ${g_def} -l ${l_def} -L ${L_def} -Q ${q_def} -a ${a} -O ${resectpath}/NIFTI/Case${c}/Case${c}-deeds_alone -S ${resectpath}/NIFTI/Case${c}/Case${c}-MRI-landmarks.nii.gz) 

  echo "Calculating TRE for deformable registration"
  cd ..
  defTRE=$(python landmarks_centre_mass_SSC.py --inputnii ${resectpath}/NIFTI/Case${c}/Case${c}-US-landmarks.nii.gz --movingnii ${resectpath}/NIFTI/Case${c}/Case${c}-deeds_alone_deformed_seg.nii.gz --savetxt ${resectpath}/NIFTI/Case${c}/Case${c}-results-deeds_alone)

  #do a linear fit
  cd deeds
  defaffTime=$(./linearBCV -F ${resectpath}/NIFTI/Case${c}/Case${c}-US.nii.gz -M ${resectpath}/NIFTI/Case${c}/Case${c}-deeds_alone_deformed.nii.gz  -G ${g_aff} -l ${l_aff} -L ${L_aff} -Q ${q_aff}  -S ${resectpath}//NIFTI/Case${c}/Case${c}-deeds_alone_deformed_seg.nii.gz -R 1 -O ${resectpath}/NIFTI/Case${c}/affine${c}_deeds_alone) 

  echo "Calculating TRE for nonlinear + linear fit"
  cd ..
  defaffTRE=$(python landmarks_centre_mass_SSC.py --inputnii ${resectpath}/NIFTI/Case${c}/Case${c}-US-landmarks.nii.gz --movingnii ${resectpath}/NIFTI/Case${c}/affine${c}_deeds_alone_deformed_seg.nii.gz --savetxt ${resectpath}/NIFTI/Case${c}/Case${c}-results-deeds_lin)

echo ${c} >> ${outfile}
echo ${iniTRE} >> ${outfile}
echo ${defTRE} >> ${outfile}
echo ${defaffTRE} >> ${outfile}
echo ${defTime} >> ${outfile}
echo ${defaffTime} >> ${outfile}

done  
  
exit

