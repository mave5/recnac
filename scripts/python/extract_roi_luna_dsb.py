import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import cross_validation
import datetime
#from skimage import measure
import utils
import cv2
import random
import time
import os
#from keras.utils import np_utils
import matplotlib.pylab as plt
#import scipy as sp
import h5py    
#import models
#import hashlib
#from keras.preprocessing.image import ImageDataGenerator
#from image import ImageDataGenerator
#%%

root_data='/media/mra/win71/data/misc/kaggle/datascience2017/data/'
path2dsb=root_data+'dsb.hdf5'
path2dsbtest=root_data+'dsbtest.hdf5'

# non zero nodes cropped
pathdsb_nz_roi=root_data+'dsb_nz_roi.hdf5'
pathdsb_byid_roi=root_data+'dsb_byid_roi.hdf5'

path2csv=root_data+'stage1_labels.csv'
path2logs='./output/logs/'

# path to nodes
path2dsbnoduls=root_data+'nfolds_dsb_nodes.hdf5'
path2dsbtest_nodes=root_data+'nfolds_dsbtest_nodes.hdf5'
pathdsbtest_roi=root_data+'dsbtest_roi.hdf5'

# spacing file
path2spacing='/media/mra/win71/data/misc/kaggle/datascience2017/data/dsb_spacing.hdf5'


# path to luna
path2luna='/media/mra/win71/data/misc/kaggle/datascience2017/LUNA2016/'
path2luna_roi=root_data+'luna_roi.hdf5'

#%%

# original size
H,W=512,512

# batch size
bs=32

# input channel to segmentation network
c_in=7

# trained data dimesnsion
h,w=256,256

# input channel to classification
z=7
hc,wc=64,64

# exeriment name to record weights and scores
experiment='luna_dsbnc_cnn_classify_nfolds'+'roi_hw_'+str(h)+'by'+str(w)+'_cin'+str(c_in)+'_z'+str(z)
print ('experiment:', experiment)

# seed point
seed = 2016
seed = np.random.randint(seed)

# checkpoint
weightfolder='./output/weights/'+experiment
if  not os.path.exists(weightfolder):
    os.makedirs(weightfolder)
    print ('weights folder created')

# number of outputs
nb_output=1


# log
now = datetime.datetime.now()
info='log_nfolds_luna_dsbnc'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
# Direct the output to a log file and to screen
loggerFileName = os.path.join(path2logs,  suffix + '.txt')
utils.initialize_logger(loggerFileName)

#%%


# crop ROI
def crop_nodule_roi(X,Y1,y,reshape_ena=True):
    if len(X.shape)==2:
        X=X[np.newaxis,np.newaxis]
    elif len(X.shape)==3:
        X=X[np.newaxis]

    if Y1 is None:
        Y1=np.zeros_like(X,dtype='uint8')

    N,C0,H0,W0=X.shape
    #hc,wc=64,64
    
    Xc=np.zeros((N,C0,hc,wc),dtype=X.dtype)
    Yc1=np.zeros((N,C0,hc,wc),dtype='uint8')

    for k in range(N):
        c,r,_=y[k,:]
        #print r,c
        #print r,c
        if r>=hc/2 and (r+hc/2)<=H0:
            r1=int(r-hc/2)
            r2=int(r+hc/2)        
        elif r<hc/2:
            r1=0
            r2=int(r1+hc)
        elif (r+hc/2)>H0:
            r2=H0
            r1=r2-hc
            

        if c>=wc/2 and (c+wc/2)<=W0:
            c1=int(c-wc/2)
            c2=int(c+wc/2)        
        elif c<wc/2:
            c1=0
            c2=int(c1+wc)
        elif (c+wc/2)>W0:
            c2=W0
            c1=c2-wc
            
            
        #print k,c2-c1,r2-r1,r2,c2,X.shape
        Xc[k,:]=X[k,:,r1:r2,c1:c2]
        Yc1[k,:]=Y1[k,:,r1:r2,c1:c2]

    return Xc,Yc1



def mask2coord(Y):
    Y=np.array(Y>0.5,dtype='uint8')
    if len(Y.shape)==2:
        Y=Y[np.newaxis,np.newaxis]
    elif len(Y.shape)==3:
        Y=Y[:,np.newaxis]

    N,C,H,W=Y.shape
        
    coords=np.zeros((N,3))
    for k in range(N):
        coords[k,:]=getRegionFromMap(Y[k,0])
    
    return coords


def getRegionFromMap(Y):
    Y=np.array(Y>0.5,'uint8')
    im2, contours, hierarchy = cv2.findContours(Y,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    
    areaArray=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaArray.append(area)
    # check for any contour    
    if len(areaArray)>0:    
        largestcontour=contours[np.argmax(areaArray)]
        (x,y),radius = cv2.minEnclosingCircle(largestcontour)
        #print x,y,radius
    else:
        (x,y),radius=(np.random.randint(H,size=2)),20
        #print x,y,radius
        #raise IOError
    return x,y,radius



def extract_dsb(ids,dataset='dsb'):

    # pre-processing 
    param_prep0={
        'h': H,
        'w': W,
        'crop'    : None,
        'norm_type' : 'minmax_bound',
        'output' : 'mask',
    }

    
    if dataset=='dsb':
        # read dsb data
        f2=h5py.File(path2dsb,'r')
    
        # read dsb nodules    
        ff_dsb_nodes=h5py.File(path2dsbnoduls,'r')
        
        # spacing x,y        
        ff_spacing=h5py.File(path2spacing,'r')
    else:    
        # read dsb data
        f2=h5py.File(path2dsbtest,'r')
    
        # read dsb nodules    
        ff_dsb_nodes=h5py.File(path2dsbtest_nodes,'r')

    # labels
    y=[]
    # image and segmentation
    XY=[]
    
    for k,id in enumerate(ids):    

        # obtain images        
        X1=f2[id]
        y1=f2[id].attrs['cancer']
        y.append(y1)
        
        print k,id,y1
        
        # obtain nodules 
        Yp_seg=ff_dsb_nodes[id]['Y']
        nzYi=np.array(ff_dsb_nodes[id]['nzYi'])[0]

        step=3 
        XY0=[]               
        for t in nzYi:   
            ind=t*step+step
            X0=X1[ind-z/2:ind+z/2+1]
            X0=utils.preprocess(X0[np.newaxis],param_prep0)

            # nodule mask
            # return to original size
            Y0=cv2.resize(Yp_seg[t,0].astype('uint8'), (W, H), interpolation=cv2.INTER_CUBIC)                
            #X0=np.append(X0,Y0[np.newaxis,np.newaxis],axis=1)
            coords=mask2coord(Y0)
            X0,_=crop_nodule_roi(X0,None,coords)
            XY0.append(X0)
            
        XY0=np.vstack(XY0)
        #print XY0.shape
        XY.append(XY0)
    XY=np.vstack(XY)    
    f2.close()
    ff_dsb_nodes.close()
    return XY,np.array(y, dtype=np.uint8)
    #return X,np.array(y, dtype=np.uint8)


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

y_nc=np.zeros(len(non_cancer),'uint8')
y_c=np.ones(len(cancer),'uint8')

# train val split
path2trainvalindx=weightfolder+'/train_val_index.npz'
if not os.path.exists(path2trainvalindx):
    trn_nc, val_nc, trn_ync, val_ync = cross_validation.train_test_split(non_cancer,y_nc, random_state=420, stratify=y_nc,test_size=0.1)                                                                   
    #trn_c, val_c, trn_yc, val_yc = cross_validation.train_test_split(cancer,y_c, random_state=420, stratify=y_c,test_size=0.2) 

    # indices of train and validation
    #trn_ids=np.concatenate((trn_nc,trn_c))
    #trn_y=np.concatenate((trn_ync,trn_yc))
    trn_ids=trn_nc
    trn_y=trn_ync


    #val_ids=np.concatenate((val_nc,val_c))
    #val_y=np.concatenate((val_ync,val_yc))
    val_ids=val_nc
    val_y=val_ync

    # shuffle
    #trn_ids,trn_y=utils.unison_shuffled_copies(trn_ids,trn_y)
    #val_ids,val_y=utils.unison_shuffled_copies(val_ids,val_y)
    np.savez(path2trainvalindx,trn_ids=trn_ids,val_ids=val_ids,trn_y=trn_y,val_y=val_y)
    print 'train validation indices saved!'    
else:
    f_trvl=np.load(path2trainvalindx)    
    trn_ids=f_trvl['trn_ids']
    trn_y=f_trvl['trn_y']    
    val_ids=f_trvl['val_ids']
    val_y=f_trvl['val_y']
    print 'train validation indices loaded!'

#%%
# dsb train data

if os.path.exists(pathdsb_nz_roi):
    print 'hdf5 exists!'
else:
    # prepare dsb data
    X_trn_nc,y_trn_nc=extract_dsb(trn_ids,dataset='dsb')
    y_trn_nc=np.zeros(X_trn_nc.shape[0],'uint8')
    ff_dsb_nz_roi=h5py.File(pathdsb_nz_roi,'w-')
    ff_dsb_nz_roi['X_train']=X_trn_nc
    ff_dsb_nz_roi['y_train']=y_trn_nc
    
    X_val_nc,y_val_nc=extract_dsb(val_ids,dataset='dsb')
    y_c=np.zeros(X_val_nc.shape[0],'uint8')
    ff_dsb_nz_roi['X_test']=X_val_nc
    ff_dsb_nz_roi['y_test']=y_val_nc   
    
    X_c,y_c=extract_dsb(cancer,dataset='dsb')
    y_c=np.ones(X_c.shape[0],'uint8')
    ff_dsb_nz_roi['X_c']=X_c
    ff_dsb_nz_roi['y_c']=y_c
    
    ff_dsb_nz_roi.close()

# verfiy
ff_dsb_nz_roi=h5py.File(pathdsb_nz_roi,'r')
print ff_dsb_nz_roi['X_train'].shape
print ff_dsb_nz_roi['y_train'].shape

print ff_dsb_nz_roi['X_c'].shape
print ff_dsb_nz_roi['y_c'].shape

#%%

# dsb test data
df_test = pd.read_csv('./output/submission/stage1_submission.csv')

# dsb test
if os.path.exists(pathdsbtest_roi):
    print 'hdf5 exists!'
else:
    ff_dsbtest_roi=h5py.File(pathdsbtest_roi,'w-')        
    for id1 in df_test.id:
        print id1
        # prepare dsb data
        X_t,y_t=extract_dsb([id1],dataset='dsbtest')
        ff_dsbtest_roi[id1]=X_t
    ff_dsbtest_roi.close()        

# verify
ff_dsbtest_roi=h5py.File(pathdsbtest_roi,'r')                
print ff_dsbtest_roi.keys()

#%%

def extract_luna_roi(X,Y):
    N,C,H,W=X.shape

    Xc=np.zeros((N,z,hc,wc))
    for k in range(X.shape[0]):
        print k
        coords=mask2coord(Y[k])
        X0,_=crop_nodule_roi(X[k],None,coords)
        X0=utils.preprocess(X0,param_prep0)
        Xc[k]=X0[0,:]    
    return Xc
        
# luna data
foldnm=0

# pre-processing 
param_prep0={
    'h': H,
    'w': W,
    'crop'    : None,
    'norm_type' : 'minmax_bound',
    'output' : 'mask',
}

path2luna_train_test=path2luna+'fold'+str(foldnm)+'_train_test.hdf5'

if os.path.exists(path2luna_roi):
    print 'hdf5 exist!'
else:
    if os.path.exists(path2luna_train_test):
        ff_luna=h5py.File(path2luna_train_test,'r')
        X_train=ff_luna['X_train']
        Y_train=ff_luna['Y_train']
        X_test=ff_luna['X_test']
        Y_test=ff_luna['Y_test']
        print 'hdf5 loaded '
    else:
        raise IOError    

    X_train=extract_luna_roi(X_train,Y_train)
    y_train=np.ones(len(X_train),'uint8')
    X_test=extract_luna_roi(X_test,Y_test)
    y_test=np.ones(len(X_test),'uint8')

    # write luna roi
    ff_luna_roi=h5py.File(path2luna_roi,'w-')    
    ff_luna_roi['X_train']=X_train
    ff_luna_roi['y_train']=y_train
    
    ff_luna_roi['X_test']=X_test
    ff_luna_roi['y_test']=y_test

    ff_luna_roi.close()        


# verfiy
ff_luna_roi=h5py.File(path2luna_roi,'r')
print ff_luna_roi['X_train'].shape
print ff_luna_roi['y_train'].shape

print ff_luna_roi['X_test'].shape
print ff_luna_roi['y_test'].shape
#%%

# dsb by id 

# dsb train
if os.path.exists(pathdsb_byid_roi):
    print 'hdf5 exists!'
else:
    ff_dsbbyid_roi=h5py.File(pathdsb_byid_roi,'w-')        
    for id1 in df_train.id:
        print id1
        # prepare dsb data
        X_t,y_t=extract_dsb([id1],dataset='dsb')
        ff_dsbbyid_roi[id1]=X_t
    ff_dsbbyid_roi.close()        

# verify
ff_dsbbyid_roi=h5py.File(pathdsb_byid_roi,'r')                
print ff_dsbbyid_roi.keys()


