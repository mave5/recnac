## Notes
---------


### misc
* nodule coordinates: (r,c,radius) or (y,x,radius)




#### LUNA2016 dataset
* data collected from various ct manufactures
* slice thickness from 0.6 to 5 mm
* data is in 10 dirs namely: subset0 to subset9 for n-fold cross-validation
* data are in .raw format
* nodule annotations are in CSVFILES/annotations.csv
* initially 1018 CT scans, excluded thick and inconsistent slices to get 888 scans
* total 888 CT scans: (89,89,89,89,89,89,89,89,88,88)=888





#### DSB 2017 dataset
* Total data: 1595 subjects
* 1397 subjects in trainig set, there is label (0/1) for them
* 198 subjects in test set with no label to be predicted
* around 26 percent have cancer
* number of slices per subject: min: 94, max: 541, avg: 179, std:67.0
* average execution time for dicom to numpy: 10 sec/subject
* average execution time for lung mask extraction: 50 sec/subject




#### Notes from Booz Allen Hamilton tutorial
* Used LUNA 2016 dataset for nodule segmentation
* simpleITK package for image analysis
* Luna 2016: file annotations.csv contains nodule coordiates and diameter
* some patients have multiple nodules listed in annotations.csv
* find the largest nodule in the patient scan
* data are .raw file and using simpleitk gives pixels hu values.
* coordinate system defined in .mhd file
* convert voxel location to real world coordiante system
* three slices at the nodule center are picked for each patient
* using theresholding and Kmeans to detect and extract the ROI, i.e., the lung
* using erosion and dialation to fill in the holes
* train U-net on ROI images and corresponding mask for segmentation
* create a list of features for feature engineering
* classifier for cancer detection: Random Forest and XGBoost
* python code: LUNA_mask_extraction.py for converting images and nodules masks into npy files
  numpy file names format: images_xxxx_yyyy.npy, where xxxx: patient number, yyyy: slice number
* run 20 epochs with a training set size of 320 and batch size of 2 in about an hour. We started obtaing reasonable nodule mask predictions after about 3 hours of training once the reported loss value approached 0.3.
* LUNA_mask_extraction.py: obtain images_sbnb_slnb.npy and nodules masks_sunb_slnb.npy as numpy files,
  where, sunb is subject number and slnb is slice number.
* nodule masks are created from x,y,z coordinates and diameter values in CSVFILES/annotations.csv 
* patients might have more than one nodule listed in annotations.csv file, as such, althout thre are 888 ct scans,
  total images collected from all subsets: 1186.
  


