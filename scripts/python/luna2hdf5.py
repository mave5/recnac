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
import dicom
import h5py
import random

# get package versions
def get_version(*vars):
    for var in vars:
        module = __import__(var)    
        print '%s: %s' %(var,module.__version__)
    
# package version    
get_version('numpy','matplotlib','cv2','sklearn','skimage','scipy','pandas')

#%%
path2luna="./output/numpy/luna/"
working_path = path2luna+"allsubsets/"
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
## load Luna
X_train = joblib.load(working_path+"trainX.joblib")
Y_train = joblib.load(working_path+"trainY.joblib")

X_test = joblib.load(working_path+"testX.joblib")
Y_test = joblib.load(working_path+"testY.joblib")

array_stats(X_train)
array_stats(Y_train)

array_stats(X_test)
array_stats(Y_test)

# reshape
#N,C,H,W=X_train.shape
#X_train=np.reshape(X_train,(N/3,3,H,W))
#Y_train=np.reshape(Y_train,(N/3,3,H,W))

#N,C,H,W=X_test.shape
#X_test=np.reshape(X_test,(N/3,3,H,W))
#Y_test=np.reshape(Y_test,(N/3,3,H,W))

#array_stats(X_train)
#array_stats(Y_train)

#array_stats(X_test)
#array_stats(Y_test)


## save as hdf5
f1=h5py.File(path2luna+'luna.hdf5','w-')

f1['X_train']=X_train
f1['X_test']=X_test

f1['Y_train']=Y_train
f1['Y_test']=Y_test


f1.close()
print 'hdf5 saved!'
