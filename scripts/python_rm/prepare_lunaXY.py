#%% nodule segmentation for LUNA dataset
import numpy as np
import cv2
import time
import os
import matplotlib.pylab as plt
#from skimage import measure
#import models
import utils
#from keras import backend as K
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
import h5py
import random
from glob import glob
#from image import ImageDataGenerator
import ntpath
#%%

# path to dataset
path2luna_external="/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/"
path2subsets=path2luna_external+"subsets/"
path2lungs=path2luna_external+"subsets_lung/"
path2luna_internal="/media/mra/win71/data/misc/kaggle/datascience2017/LUNA2016/"

subset_list=glob(path2subsets+'subset*.hdf5')
subset_list.sort()
print 'total subsets: %s' %len(subset_list)

#%%

# original data dimension
H = 512
W = 512

# pre-processed data dimesnsion
z,h,w=1,512,512


########## log
import datetime
path2logs='./output/logs/'
now = datetime.datetime.now()
info='log_LunaLungSegmentation_'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
# Direct the output to a log file and to screen
loggerFileName = os.path.join(path2logs,  suffix + '.txt')
utils.initialize_logger(loggerFileName)


#%%

# load train data
def load_data(subsets):
    X=[]
    Y=[]
    for ss in subsets:
        print ss
        ff_X=h5py.File(ss,'r')
        ss_lung=ss.replace('subsets','subsets_lung')        
        ss_lung=ss_lung.replace('.hdf5','_lung.hdf5')        
        print ss_lung       
        ff_Y=h5py.File(ss_lung,'r')
        for k in ff_X.keys():
            print k
            X0=ff_X[k][0]
            Y0=ff_Y[k]
            #print X0.shape,X0.dtype
            #print Y0.shape,Y0.dtype
            rnd_inds=random.sample(xrange(X0.shape[0]), int(0.1*X0.shape[0]))
            rnd_inds.sort()
            #print rnd_inds
            X.append(X0[rnd_inds])
            Y.append(Y0[rnd_inds])
        ff_X.close()    
        ff_Y.close()    
    X=np.vstack(X)    
    print X.shape
    Y=np.vstack(Y)
    print Y.shape
    return X,Y

#%%

print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

    
# path to nfold train and test data
path2_lunaXY=path2luna_external+'luna_XY.hdf5'
if os.path.exists(path2_lunaXY):
    print 'hdf5 exists!'
else:    
    
    # load images and masks 
    ff_w=h5py.File(path2_lunaXY,'w-')
    
    for ss in subset_list:
        print ss
        print ntpath.basename(ss)[:-5]
        
        # create group for subset
        grp_ss=ff_w.create_group(ntpath.basename(ss[:-5]))
        
        # read images        
        ff_X=h5py.File(ss,'r')
        # read lungs
        ss_lung=ss.replace('subsets','subsets_lung')        
        ss_lung=ss_lung.replace('.hdf5','_lung.hdf5')        
        print ss_lung       
        ff_Y=h5py.File(ss_lung,'r')
        
        for key in ff_X.keys():
            print 'subject:', key
            X=ff_X[key][0]
            Ylung=ff_Y[key].value
            print X.shape, Ylung.shape,X.dtype,Ylung.dtype
            
            # create group for series id
            grp_key=grp_ss.create_group(key)
            grp_key['X']=X
            grp_key['Y']=Ylung
            
        print '-'*50    
        ff_X.close()    
        ff_Y.close()    
    ff_w.close()    
#%%
# verify        
nX,nY=0,0
ff_w=h5py.File(path2_lunaXY,'r')
print 'subsets:', ff_w.keys()
for key in ff_w.keys():
    print key
    for key2 in ff_w[key].keys():
        print key2,ff_w[key][key2].keys()
        print ff_w[key][key2]['X'].shape,ff_w[key][key2]['Y'].shape
        nX=nX+ff_w[key][key2]['X'].shape[0]
        nY=nY+ff_w[key][key2]['Y'].shape[0]        
        print ff_w[key][key2]['X'].dtype,ff_w[key][key2]['Y'].dtype
        
print 'total images:', nX,nY        