## Notes
---------

### DSB 2017 dataset
* Total data: 1595 subjects
* 1397 subjects in trainig set, there is label (0/1) for them
* 198 subjects in test set with no label to be predicted
* around 26 percent have cancer
* number of slices per subject: min: 94, max: 541, avg: 179, std:67.0





#### Notes from Booz Allen Hamilton tutorial
* Used LUNA 2016 dataset for nodule segmentation
* simpleITK package for image analysis
* Luna 2016: file annotations.csv contains nodule coordiates and diameter
* some patients have multiple nodules listed in annotations.csv
* find the largest nodule in the patient scan
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




