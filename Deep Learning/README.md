Model Training: Run the code in the Model Training python notebook. This uses all the parameters set in the paired_mrus_brain.yaml configuration file to load in the specified data, build a model with the specified parameters, and train the model with a callback on the L2 validation loss.

Model Prediction: Run the following DeepReg command from the Deep Learning directory

```
bash deepreg_predict --gpu "0" --ckpt_path logs/logs_train/20210413-172130/20210413-172130/save/ckpt-91 --mode test --exp_name 91_final_test
```

mTRE Calculation: Run the code in the mTRE Calculations python notebook. This uses the prediction results from DeepReg and recalculates the mTRE to account for the transformation from image coordinates to real-world coordinates, and to account for the resampling / scaling done during the data preprocessing. The mTRE results are printed out in the mTRE Calcuations notebook.


Note: DeepReg has been updated quite frequently and some functionality may change from when this was run. To update this to run with any newer version of DeepReg simply modify the configuration file to work with whatever updates, add the L2 loss code from this repository to deepreg/loss/label.py, and update deepreg/train.py to include the same L2 validation loss callback and training history return. 
