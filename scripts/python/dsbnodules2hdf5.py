# conver luna to hdf5

import numpy as np
import cv2
import time
from glob import glob
import os
import matplotlib.pylab as plt
from sklearn.externals import joblib
from skimage import measure
from skimage.transform import resize
import ntpath
#%matplotlib inline
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#p = sns.color_palette()
from skimage.draw import circle
#import re
#import traceback
#import dicom
import h5py
#import random

# get package versions
def get_version(*vars):
    for var in vars:
        module = __import__(var)    
        print '%s: %s' %(var,module.__version__)
    
# package version    
get_version('numpy','matplotlib','cv2','sklearn','skimage','scipy','pandas')

#%%

path2noduls='./output/numpy/dsb/'
path2output='./output/data/dsb/'
#%%

def array_stats(X):
    X=np.asarray(X)
    
    # get var name
    #stack = traceback.extract_stack()
    #filename, lineno, function_name, code = stack[-2]
    #vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    
    #print (vars_name,X.shape, X.dtype)
    print ('array shape',X.shape, X.dtype)
    #print 'min: %.3f, max:%.3f, avg: %.3f, std:%.3f' %(np.min(X),np.max(X),np.mean(X),np.std(X))
    print ('min: {}, max: {}, avg: {:.3}, std:{:.3}'.format( np.min(X),np.max(X),np.mean(X),np.std(X)))
    
    print '-'*50

#%%

# dsb nodule files
nodefile_list=glob(path2noduls+'*nodule.npz')
print ('total nodes files: %s' %(len(nodefile_list)))

f2=h5py.File(path2output+'dsb_nodes.hdf5','w-')
for i,nodfile in enumerate(nodefile_list):
    print ('processing: %s, %s' %(i,nodfile))

    subject_id=ntpath.basename(nodfile)
    subject_id=subject_id.replace("_nodule.npz","")
    
    # read node from numpy
    f1 = np.load(nodfile)
    y_node=f1['Y'] # nodule 
    
    # store into hdf5    
    f2[subject_id]=y_node
    
f2.close()
#%% verify

f3=h5py.File(path2output+'dsb_nodes.hdf5','r')
print f3.keys()
f3.close()



