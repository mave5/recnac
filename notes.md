## Notes
---------



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




