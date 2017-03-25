#from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
#import csv
import h5py
#import os
from glob import glob
#import pandas as pd
#import ntpath
#from skimage.draw import circle

try:
    from tqdm import tqdm # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x
#%%
H,W=512,512

# Getting list of image files
luna_path_save = "/media/mra/win71/data/misc/kaggle/datascience2017/LUNA2016/" 
path2lunasegs='/media/mra/win71/data/misc/kaggle/datascience2017/LUNA2016/seg-lungs-LUNA16/'
path2luna="/media/mra/win71/data/misc/kaggle/datascience2017/LUNA2016/"

#luna_csv_path = luna_path+"CSVFILES/"

# list of subsets
subset_list=glob(path2luna+'subset*.hdf5')
subset_list.sort()
print 'total subsets: %s' %len(subset_list)

    
#%%
#Some helper functions

#####################
#
# Helper function to get rows in data frame associated 
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing

# load train data
def load_data(subsets):
    Y=[]
    for ss in subsets:
        print ss
        ff=h5py.File(ss,'r')
        ff_lung=h5py.File(ss[:-5]+'_lung.hdf5','w-')
        for k,key in enumerate(ff.keys()):
           #print path2lunasegs+k 
           Y, numpyOrigin, numpySpacing = load_itk_image(str(path2lunasegs+key))
           #X = ff[k]
           print k,Y.shape
           ff_lung[key]=Y.astype('uint8')
        ff.close()    
        ff_lung.close()
        print 'subset compeleted!'
        print '-'*50


   

#%% loop over all subsets

load_data(subset_list)
print 'completed!'





