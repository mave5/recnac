#from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
#import csv
import scipy.ndimage
import h5py
import os
from glob import glob
#import pandas as pd
import ntpath
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
path2lunasubset="/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/"

#luna_csv_path = luna_path+"CSVFILES/"

# list of subsets
subset_list=glob(path2lunasubset+'subset*.hdf5')
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

## load train data
#def load_data(subsets):
#    Y=[]
#    for ss in subsets:
#        print ss
#        ff=h5py.File(ss,'r')
#        path2rescale=path2luna+ntpath.basename(ss)
#        path2rescale=path2rescale[:-5]+'_rescale.hdf5' 
#        print path2rescale
#
#        if os.path.exists(path2rescale):
#            print 'rescale subset exists!'
#            
#        else:
#            ff_Xrescale=h5py.File(path2rescale,'w-')
#            for k,key in enumerate(ff.keys()):
#               #print path2lunasegs+k 
#               Y, numpyOrigin, numpySpacing = load_itk_image(str(path2lunasegs+key))
#               X = ff[key]
#               print k,Y.shape,X.shape,numpySpacing
#               X0=resample(X[0],numpySpacing)
#               
#               print k,X0.shape,X0.dtype
#               ff_Xrescale[key]=X0
#            ff.close()    
#            ff_Xrescale.close()
#            print 'subset compeleted!'
#            print '-'*50


# load train data
def resample_lung(subsets):
    Y=[]
    for ss in subsets:
        print ss
        ff=h5py.File(ss,'r')
        path2rescale=path2luna+ntpath.basename(ss)
        path2rescale=path2rescale[:-5]+'_lung_rescale.hdf5' 
        print path2rescale

        if os.path.exists(path2rescale):
            print 'rescale subset exists!'
            
        else:
            ff_Xrescale=h5py.File(path2rescale,'w-')
            for k,key in enumerate(ff.keys()):
               #print path2lunasegs+k 
               Y, numpyOrigin, numpySpacing = load_itk_image(str(path2lunasegs+key))
               #X = ff[key]
               print k,Y.shape,np.min(Y),np.max(Y),numpySpacing
               
               # extract lung                
               Ylung=(Y==4) | (Y==3)
               # resample lung segmentation
               Ylung=resample(np.array(Ylung,'uint8'),numpySpacing)>0
               print 'resampled:',Ylung.shape,np.min(Ylung),np.max(Ylung)
               # resample image
               #X0=resample(X[0],numpySpacing)
               #X0=normalize(X0)*Ylung
               #print k,X0.shape,X0.dtype,np.min(X0),np.max(X0)
               #X0=np.asarray(X0*(2**15-1),'int16')
               #print Ylung.shape
               #print k,X0.shape,X0.dtype,np.min(X0),np.max(X0)
               
               ff_Xrescale[key]=np.array(Ylung,'uint8')
            ff.close()    
            ff_Xrescale.close()
            print 'subset compeleted!'
            print '-'*50


def resample(image, spacing, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    #spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    #new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image

def normalize(image):
    ####### normalization
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image
#%% loop over all subsets

#load_data(subset_list)
resample_lung(subset_list)
print 'completed!'





