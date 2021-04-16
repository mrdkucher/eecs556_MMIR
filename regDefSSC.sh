#!/bin/bash

mmirpath="/home/yeatsem/Proj556/eecs556_MMIR" #put the path to the local version of the github repo here
itkpath="/home/yeatsem/itk/bin" #put the path to itk binary folder here
resectpath="/home/yeatsem/Proj556/RESECT" #put the path to RESECT data here
deedspath="/home/yeatsem/Proj556/deeds" #put path to the deeds binaries here

outfile="/home/yeatsem/Proj556/DefAlone-4Levels-Results.txt"

g="4x3x2x1" #grid spacing
l="10x9x7x6" #maximum search radius
q="4x3x2x1" #quantization
  
for c in {1..27}
do
  #preprocess the RESECT data
  cd ${itkpath}
  ./c3d ${resectpath}/NIFTI/Case${c}/US/Case${c}-US-before.nii.gz ${resectpath}/NIFTI/Case${c}/MRI/Case${c}-FLAIR.nii.gz -reslice-identity -resample-mm 0.5x0.5x0.5mm -o ${resectpath}/NIFTI/Case${c}/Case${c}-MRI_in_US.nii.gz

  ./c3d ${resectpath}/NIFTI/Case${c}/US/Case${c}-US-before.nii.gz -resample-mm 0.5x0.5x0.5mm -o ${resectpath}/NIFTI/Case${c}/Case${c}-US.nii.gz

  #save landmark coordinates as .txt
  cd ${mmirpath}
  python landmarks_split_txt.py --inputtag ${resectpath}/NIFTI/Case${c}/Landmarks/Case${c}-MRI-beforeUS.tag --savetxt ${resectpath}/NIFTI/Case${c}/Case${c}_lm
  
  #save landmark coordinates as .txt
  cd ${mmirpath}
  python landmarks_split_txt.py --inputtag ${resectpath}/NIFTI/Case${c}/Landmarks/Case${c}-MRI-beforeUS.tag --savetxt ${resectpath}/NIFTI/Case${c}/Case${c}_lm

  #generate MRI voxelized landmarks
  cd ${itkpath}
  ./c3d ${resectpath}/NIFTI/Case${c}/Case${c}-MRI_in_US.nii.gz -scale 0 -landmarks-to-spheres ${resectpath}/NIFTI/Case${c}/Case${c}_lm_mri.txt 1 -o ${resectpath}/NIFTI/Case${c}/Case${c}-MRI-landmarks.nii.gz

  # generate US voxelized landmarks
  ./c3d ${resectpath}/NIFTI/Case${c}/Case${c}-US.nii.gz -scale 0 -landmarks-to-spheres ${resectpath}/NIFTI/Case${c}/Case${c}_lm_us.txt 1 -o ${resectpath}/NIFTI/Case${c}/Case${c}-US-landmarks.nii.gz

  echo "Calculating initial TRE"
  cd ${mmirpath}
  iniTRE=$(python landmarks_centre_mass.py --inputnii ${resectpath}/NIFTI/Case${c}/Case${c}-US-landmarks.nii.gz --movingnii ${resectpath}/NIFTI/Case${c}/Case${c}-MRI-landmarks.nii.gz --savetxt ${resectpath}/NIFTI/Case${c}/Case${c}-prelim)
  
  # run deformable deeds
  cd ${deedspath}
  ./deedsBCV -F ${resectpath}/NIFTI/Case${c}/Case${c}-US.nii.gz -M ${resectpath}/NIFTI/Case${c}/Case${c}-MRI_in_US.nii.gz  -G ${g} -L ${l} -Q ${q} -O ${resectpath}/NIFTI/Case${c}/Case${c}-deeds_alone -S ${resectpath}/NIFTI/Case${c}/Case${c}-MRI-landmarks.nii.gz 

  echo "Calculating TRE for deformable registration"
  cd ${mmirpath}
  defTRE=$(python landmarks_centre_mass.py --inputnii ${resectpath}/NIFTI/Case${c}/Case${c}-US-landmarks.nii.gz --movingnii ${resectpath}/NIFTI/Case${c}/Case${c}-deeds_alone_deformed_seg.nii.gz --savetxt ${resectpath}/NIFTI/Case${c}/Case${c}-results-deeds_alone)

  #do a linear fit
  cd ${deedspath}
  ./linearBCV -F ${resectpath}/NIFTI/Case${c}/Case${c}-US.nii.gz -M ${resectpath}/NIFTI/Case${c}/Case${c}-deeds_alone_deformed.nii.gz -S ${resectpath}/NIFTI/Case${c}/Case${c}-deeds_alone_deformed_seg.nii.gz -R 1 -O ${resectpath}/NIFTI/Case${c}/affine${c}_deeds_alone 

  echo "Calculating TRE for nonlinear + linear fit"
  cd ${deedspath} 
  defaffTRE=$(python landmarks_centre_mass.py --inputnii resectpath}/NIFTI/Case${c}/Case${c}-US-landmarks.nii.gz --movingnii resectpath}/NIFTI/Case${c}/affine${c}_deeds_alone_deformed_seg.nii.gz --savetxt resectpath}/NIFTI/Case${c}/Case${c}-results-deeds_lin)

echo ${c} >> ${outfile}
echo ${iniTRE} >> ${outfile}
echo ${defTRE} >> ${outfile}
echo ${defaffTRE} >> ${outfile}

done  
  
exit
