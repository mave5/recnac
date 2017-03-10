import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from sklearn import cross_validation
import datetime
import utils
#import cv2
#import random
#import time
import os
import matplotlib.pylab as plt
import scipy as sp
import h5py    
import models
import hashlib
#from keras.preprocessing.image import ImageDataGenerator
#from image import ImageDataGenerator
#%%
path2luna="/media/mra/win7/data/misc/kaggle/datascience2017/LUNA2016/"
root_data='/media/mra/win7/data/misc/kaggle/datascience2017/data/'
path2dsb=root_data+'dsb.hdf5'
path2dsbtest=root_data+'dsbtest.hdf5'

path2csv=root_data+'stage1_labels.csv'
path2logs='./output/logs/'

# resize
H,W=512,512

# batch size
bs=8

c_in=7

# trained data dimesnsion
h,w=256,256

# time step
z=2

# exeriment name to record weights and scores
experiment='dsb_cnn_classify'+'roi_hw_'+str(h)+'by'+str(w)+'_cin'+str(c_in)+'_z'+str(z)
print ('experiment:', experiment)

# seed point
seed = 2016
seed = np.random.randint(seed)

# checkpoint
weightfolder='./output/weights/'+experiment
if  not os.path.exists(weightfolder):
    #os.makedirs(weightfolder)
    print ('weights folder does not exist!')
    raise IOError

# number of outputs
nb_output=1

# fast train
fast_train=False

# log
now = datetime.datetime.now()
info='log'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
# Direct the output to a log file and to screen
loggerFileName = os.path.join(path2logs,  suffix + '.txt')
utils.initialize_logger(loggerFileName)

# loading pre-train weights
pre_train=True 

#%%

print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

########## load DSB data, only non-cancer
df_train = pd.read_csv(path2csv)
print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
df_train.head()

# extract non cancers
non_cancer=df_train[df_train.cancer==0].id
cancer=df_train[df_train.cancer==1].id
print 'total non cancer:%s, total cancer:%s' %(len(non_cancer),len(cancer))

#%%

# nodule segmentation network

# training params
params_seg={
    'h': h,
    'w': w,
    'c_in': c_in,           
    'weights_path': None,        
    'learning_rate': 3e-5,
    'optimizer': 'Adam',
    #'loss': 'binary_crossentropy',
    #'loss': 'mean_squared_error',
    'loss': 'dice',
    'nbepoch': 1000,
    'c_out': 1,
    'nb_filters': 16,    
    'max_patience': 30    
        }

seg_model=models.seg_model(params_seg)

seg_model.summary()

# path to weights
path2segweights=weightfolder+"/weights_seg.hdf5"
if  os.path.exists(path2segweights):
    seg_model.load_weights(path2segweights)
    print 'weights loaded!'
else:
    raise IOError
    
weights_checksum=hashlib.md5(open(path2segweights, 'rb').read()).hexdigest()
print 'checksum:', weights_checksum
with open(weightfolder+'/checksum.csv', 'w+') as f:
    f.write(weights_checksum + '\n')    
    
#%%
    
# verify network
    
# pre-processing 
param_prep={
    'h': h,
    'w': w,
    'crop'    : None,
    'norm_type' : 'minmax_bound',
    'output' : 'mask',
}

path2luna_train_test=path2luna+'train_test.hdf5'
if os.path.exists(path2luna_train_test):
    ff_r=h5py.File(path2luna_train_test,'r')
    X_test=ff_r['X_test']
    Y_test=ff_r['Y_test']
    print 'hdf5 loaded '

X,Y=utils.preprocess_XY(X_test,Y_test,param_prep)

score_test=seg_model.evaluate(*utils.preprocess_XY(X_test,Y_test,param_prep),verbose=0)
print ('score_test: %.2f' %(score_test[1]))

#%%
# obtain nodules for dsb train

path2dsboutput='./output/data/dsb/dsb_nodules.hdf5'

if not os.path.exists(path2dsboutput):
    f3=h5py.File(path2dsboutput,'w-')

    # read dsb hdf5
    f2=h5py.File(path2dsb,'r')

    for id in df_train.id:
        print id
        
        X1=f2[id]
        
        n=X1.shape[0]
        X1=np.append(X1,np.zeros((c_in-n%c_in,H,W),dtype='int16'),axis=0)
        #print X1.shape
        n=X1.shape[0]
        X1=np.reshape(X1,(n/c_in,c_in,H,W))
        #print X1.shape
        #X.append(X1)
        y1=f2[id].attrs['cancer']
    
        # preprocess        
        X1=utils.preprocess(X1,param_prep)
             
        # obtain nodules 
        Yp_seg=seg_model.predict(X1)>.5
    
        f3[id]=Yp_seg
        
    f3.close()
else:
    print 'dsb nodules exist!'
    
#%%

# obtai nodules for dsb test

path2dsbtest_output='./output/data/dsb/dsbtest_nodules.hdf5'

if not os.path.exists(path2dsbtest_output):
    f4=h5py.File(path2dsbtest_output,'w-')

    # read dsb hdf5
    f5=h5py.File(path2dsbtest,'r')

    for k,id in enumerate(f5.keys()):
        print k, id
        
        X1=f5[id]
        
        n=X1.shape[0]
        X1=np.append(X1,np.zeros((c_in-n%c_in,H,W),dtype='int16'),axis=0)
        #print X1.shape
        n=X1.shape[0]
        X1=np.reshape(X1,(n/c_in,c_in,H,W))
        #print X1.shape
        #X.append(X1)
        y1=f5[id].attrs['cancer']
    
        # preprocess        
        X1=utils.preprocess(X1,param_prep)
             
        # obtain nodules 
        Yp_seg=seg_model.predict(X1)>.5
    
        f4[id]=Yp_seg
        
    f4.close()
else:
    print 'dsb test nodules exist!'
    


