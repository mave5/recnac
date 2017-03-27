#%% nodule segmentation for LUNA dataset
import numpy as np
import cv2
import time
import os
import scipy.ndimage
import matplotlib.pylab as plt
#from skimage import measure
import models
import utils
from keras import backend as K
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
import h5py
import random
from glob import glob
from keras.models import load_model
#from image import ImageDataGenerator

#%%

# path to dataset
path2luna_external="/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/"
path2luna_XY=path2luna_external+'luna_XY.hdf5'
ff_luna_XY=h5py.File(path2luna_XY,'r')
subset_list=ff_luna_XY.keys()
subset_list.sort()
print 'total subsets: %s' %len(subset_list)


path2dsb_external="/media/mra/My Passport/Kaggle/datascience2017/hpz440/data/hdf5/"
path2dsb_internal="/media/mra/win71/data/misc/kaggle/datascience2017/data/"

#%%

# original data dimension
H = 512
W = 512

# pre-processed data dimesnsion
z,h,w=1,512,512

# image and label channels
c_out=1

# batch size
bs=16

# fold number
foldnm=1

# exeriment name to record weights and scores
experiment='fold'+str(foldnm)+'_luna_lung_seg'+'_hw_'+str(h)+'by'+str(w)+'_cin'+str(z)
print ('experiment:', experiment)

# seed point
seed = 2017
seed = np.random.randint(seed)

# checkpoint
weightfolder='./output/weights/'+experiment
if  not os.path.exists(weightfolder):
    os.makedirs(weightfolder)
    print ('weights folder created')

# data augmentation 
augmentation=False

# fast train
pre_train=False

########## log
import datetime
path2logs='./output/logs/'
now = datetime.datetime.now()
info='log_DeployLungSeg_'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
# Direct the output to a log file and to screen
loggerFileName = os.path.join(path2logs,  suffix + '.txt')
utils.initialize_logger(loggerFileName)


#%%

# load train data
def load_data(subsets,display=False):
    X=[]
    Y=[]
    for ss in subsets:
        if display:
            print ss
        for key in ff_luna_XY[ss].keys():
            if display:
                print key
            X0=ff_luna_XY[ss][key]['X']
            Y0=ff_luna_XY[ss][key]['Y']
            rnd_inds=random.sample(xrange(X0.shape[0]), 10)
            rnd_inds.sort()
            if display:            
                print X0.shape,X0.dtype
                print Y0.shape,Y0.dtype
                print rnd_inds
            X.append(X0[rnd_inds])
            Y.append(Y0[rnd_inds])
    X=np.vstack(X)    
    #print X.shape
    Y=np.vstack(Y)
    #print Y.shape
    return X,Y

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

#%%

print('-'*30)
print('Loading and preprocessing test data...')
print('-'*30)

ss_test=subset_list.pop(foldnm)
print 'test:', ss_test

X_test,Y_test=load_data([ss_test])
X_test=utils.normalize(X_test)
# extract lung only
Y_test=(Y_test==3)|(Y_test==4)    
utils.array_stats(X_test)
utils.array_stats(Y_test)

    
#%%
print('-'*30)
print('Creating and compiling model...')
print('-'*30)

# training params
params_train={
    'h': h,
    'w': w,
    'z': z,           
    'weights_path': None, 
    'initial_lr': 3e-5,       
    'learning_rate': 1.5e-5,
    'optimizer': 'Adam',
    #'loss': 'binary_crossentropy',
    #'loss': 'mean_squared_error',
    'loss': 'dice',
    'nbepoch': 2000,
    'nb_output': c_out,
    'nb_filters': 16,    
    'max_patience': 50    
        }

model = models.seglung_model(params_train)
model.summary()

# path to weights
path2weights=weightfolder+"/weights.hdf5"
if os.path.exists(path2weights):
    model.load_weights(path2weights)
    print 'weights loaded!'
else:
    raise IOError('weights does not exist!')    

#%%
          
# evaluate on test and train data
score_test=model.evaluate(X_test[:,np.newaxis],Y_test[:,np.newaxis],verbose=0,batch_size=bs)
if len(score_test)>1:
    score_test=score_test[1]   
print 'score_test: %.2f' %(score_test)
print '-'*50
#%%

ff_dsb=h5py.File(path2dsb_external+'dsb.hdf5','r')
print 'total subjects:', len(ff_dsb)
path2dsb_resample=path2dsb_internal+'fold'+str(foldnm)+'_dsb_resampleXY.hdf5'

# load metadata
path2dsb_spacing=path2dsb_external+'dsb_spacing.hdf5'
ff_dsb_spacing=h5py.File(path2dsb_spacing,'r')
print 'total subjects:', len(ff_dsb_spacing)

# storing image and lung after resampling
ff_w=h5py.File(path2dsb_resample,'w-')

# loop over dsb
for k,key in enumerate(ff_dsb.keys()):
    print 'patient %s id: %s' %(k,key)
    X=ff_dsb[key].value
    
    # normalize    
    Xn=utils.normalize(X)
    
    #print X.shape
    Y_pred=model.predict(Xn[:,np.newaxis])>0.5
    Y_pred=Y_pred[:,0] # convert to shape N*H*W
    
    # get spacing
    spacing=ff_dsb_spacing[key].value
    print spacing
    
    # resample X
    X=resample(X,spacing)
    Y_pred=resample(np.array(Y_pred,'uint8'),spacing)>0
    
    # store in hdf5
    grp=ff_w.create_group(key)
    grp['X']=X
    grp['Y']=Y_pred
    print X.shape,X.dtype,Y_pred.shape,Y_pred.dtype
    
ff_w.close()    
print '-'*50
    # display
    #n1=np.random.randint(X.shape[0])
    #XY1=utils.image_with_mask(X[n1],Y_pred[n1,0])
    #plt.imshow(XY1)
    
#%%

ff_dsbtest=h5py.File(path2dsb_external+'dsbtest.hdf5','r')
print 'total subjects:', len(ff_dsbtest)
path2dsbtest_resample=path2dsb_internal+'fold'+str(foldnm)+'_dsbtest_resampleXY.hdf5'

# load metadata
path2dsbtest_spacing=path2dsb_external+'dsbtest_spacing.hdf5'
ff_dsbtest_spacing=h5py.File(path2dsbtest_spacing,'r')
print 'total subjects:', len(ff_dsbtest_spacing)

# storing image and lung after resampling
ff_w=h5py.File(path2dsbtest_resample,'w-')

# loop over dsb
for k,key in enumerate(ff_dsbtest.keys()):
    print 'patient %s id: %s' %(k,key)
    X=ff_dsbtest[key].value
    
    # normalize    
    Xn=utils.normalize(X)
    
    #print X.shape
    Y_pred=model.predict(Xn[:,np.newaxis])>0.5
    Y_pred=Y_pred[:,0] # convert to shape N*H*W
    
    # get spacing
    spacing=ff_dsb_spacing[key].value
    print spacing
    
    # resample X
    X=resample(X,spacing)
    Y_pred=resample(np.array(Y_pred,'uint8'),spacing)>0
    
    # store in hdf5
    grp=ff_w.create_group(key)
    grp['X']=X
    grp['Y']=Y_pred
    print X.shape,X.dtype,Y_pred.shape,Y_pred.dtype
    
ff_w.close()    
print '-'*50



#%%
# display
n1=np.random.randint(X.shape[0])
if np.max(X[n1])>1:
    Xn1=utils.normalize(X[n1])
XY1=utils.image_with_mask(Xn1,Y_pred[n1]>0.5)
plt.imshow(XY1)
plt.title(n1)



