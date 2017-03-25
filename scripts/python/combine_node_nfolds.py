import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import utils
#import cv2
#import random
#import time
import os
import matplotlib.pylab as plt
#import scipy as sp
import h5py    
import models
import hashlib
#from keras.preprocessing.image import ImageDataGenerator
#from image import ImageDataGenerator
#%% data path

# luna data
path2luna="/media/mra/win71/data/misc/kaggle/datascience2017/LUNA2016/"
path2luna_nodes="/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/nodes/"

# dsb data
root_data='/media/mra/win71/data/misc/kaggle/datascience2017/data/'
path2dsb=root_data+'dsb.hdf5'
path2dsbtest=root_data+'dsbtest.hdf5'

# stage1 labels
path2csv=root_data+'stage1_labels.csv'
path2dsbtestcsv=root_data+'stage1_submission.csv'

# logs
path2logs='./output/logs/'

#%%


# original size
H,W=512,512


# trained data dimesnsion
h,w=256,256

# input channel size
c_in=7

# seed point
seed = 2016
seed = np.random.randint(seed)
  
# number of outputs
nb_output=1

# log
now = datetime.datetime.now()
info='log_deploy_nfolds'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
# Direct the output to a log file and to screen
loggerFileName = os.path.join(path2logs,  suffix + '.txt')
utils.initialize_logger(loggerFileName)
#%%

print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

########## load DSB data
df_train = pd.read_csv(path2csv)
print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
df_train.head()

# extract non cancers and non-cancers
non_cancer=df_train[df_train.cancer==0].id
cancer=df_train[df_train.cancer==1].id
print 'total non cancer:%s, total cancer:%s' %(len(non_cancer),len(cancer))

df_test = pd.read_csv(path2dsbtestcsv)
print('Number of training patients: {}'.format(len(df_test)))
df_test.head()

#%%

# path to all folds output
path2nfolds_dsb=path2luna_nodes+'nfolds'+'_dsb_nodes.hdf5'
ff_w=h5py.File(path2nfolds_dsb,'w-')

# get file objects for all node hdf5
ff=[]
nb_folds=10
for foldnm in range(nb_folds):
    print 'fold ', foldnm

    # obtain nodules for dsb train
    path2dsboutput=path2luna_nodes+'/fold'+str(foldnm)+'_dsb_nodes.hdf5'

    if not os.path.exists(path2dsboutput):
        raise IOError
    else:
        ff.append(h5py.File(path2dsboutput,'r'))



for id in  df_train.id:
    print id
    
    # read first file
    Y0=ff[0][id]['Y']
    Y=np.zeros_like(Y0)
    
    # average over all folds
    for f in ff:
        print 'fold ', f
    
        Y=Y+f[id]['Y']
    Y=Y/nb_folds        
    
    # sum over masks
    sumYp=np.sum(Y>0.5,axis=(1,2,3))
    
    # number of non_zeros
    numofnzY=np.count_nonzero(sumYp)    
    
    # non-zero indices
    nzY_inds=np.nonzero(sumYp)    
    
    # sort by area
    sumYp_sort=np.argsort(-sumYp)

    print 'number of non-zero masks: %s' %numofnzY        
    
    # write average nodes into hdf5    
    grp=ff_w.create_group(id)
    grp['Y']=Y # mask
    grp['cnz']=numofnzY # number of non-zero masks
    grp['nzYi']=nzY_inds # non zero indices
    grp['sYi']=sumYp_sort # sorted indices by area

ff_w.close()

#%%


# path to all folds output
path2nfolds_dsb=path2luna_nodes+'nfolds'+'_dsbtest_nodes.hdf5'
ff_w=h5py.File(path2nfolds_dsb,'w-')

# get file objects for all node hdf5
ff=[]
nb_folds=10
for foldnm in range(nb_folds):
    print 'fold ', foldnm

    # obtain nodules for dsb train
    path2dsboutput=path2luna_nodes+'/fold'+str(foldnm)+'_dsbtest_nodes.hdf5'

    if not os.path.exists(path2dsboutput):
        raise IOError
    else:
        ff.append(h5py.File(path2dsboutput,'r'))



for id in  df_test.id:
    print id
    
    # read first file
    Y0=ff[0][id]['Y']
    Y=np.zeros_like(Y0)
    
    # average over all folds
    for f in ff:
        print 'fold ', f
    
        Y=Y+f[id]['Y']
    Y=Y/nb_folds        
    
    # sum over masks
    sumYp=np.sum(Y>0.5,axis=(1,2,3))
    
    # number of non_zeros
    numofnzY=np.count_nonzero(sumYp)    
    
    # non-zero indices
    nzY_inds=np.nonzero(sumYp)    
    
    # sort by area
    sumYp_sort=np.argsort(-sumYp)

    print 'number of non-zero masks: %s' %numofnzY        
    
    # write average nodes into hdf5    
    grp=ff_w.create_group(id)
    grp['Y']=Y # mask
    grp['cnz']=numofnzY # number of non-zero masks
    grp['nzYi']=nzY_inds # non zero indices
    grp['sYi']=sumYp_sort # sorted indices by area

ff_w.close()



