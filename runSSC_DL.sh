#!/bin/bash

datapath="/home/yeatsem/Proj556/eecs556_MMIR/Deep Learning/logs/91_final_test/test/pair_"
 #put the path to the local version of the github repo here
deedspath="/home/yeatsem/Proj556/deeds"

for p in {0..2}
do
	path="$datapath"${p} 
        
	cd ${deedspath}
        
	./linearBCV -F "${path}"/fixed_image.nii.gz -M "${path}"/pred_fixed_image.nii.gz -O "${path}"/affine 
done

exit
