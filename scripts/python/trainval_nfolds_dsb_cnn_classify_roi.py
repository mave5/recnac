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
from keras.utils import np_utils
import matplotlib.pylab as plt
#import scipy as sp
import h5py    
import models
#import hashlib
#from keras.preprocessing.image import ImageDataGenerator
from image import ImageDataGenerator
#%%

root_data='/media/mra/win71/data/misc/kaggle/datascience2017/data/'
path2dsb=root_data+'dsb.hdf5'
path2dsbtest=root_data+'dsbtest.hdf5'

path2csv=root_data+'stage1_labels.csv'
path2logs='./output/logs/'

# path to nodes
path2dsbnoduls=root_data+'nfolds_dsb_nodes.hdf5'
path2dsbtest_nodes=root_data+'nfolds_dsbtest_nodes.hdf5'

# spacing file
path2spacing='/media/mra/win71/data/misc/kaggle/datascience2017/data/dsb_spacing.hdf5'

#%%

# resize
H,W=512,512

# batch size
bs=32

# input channel to segmentation network
c_in=7

# trained data dimesnsion
h,w=256,256

# input channel to classification
z=2
hc,wc=128,128

# exeriment name to record weights and scores
experiment='dsb_cnn_classify_nfolds'+'roi_hw_'+str(h)+'by'+str(w)+'_cin'+str(c_in)+'_z'+str(z)
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

# fast train
fast_train=True

# log
now = datetime.datetime.now()
info='log_nfolds'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
# Direct the output to a log file and to screen
loggerFileName = os.path.join(path2logs,  suffix + '.txt')
utils.initialize_logger(loggerFileName)

# loading pre-train weights
pre_train=False

#%%

# random data generator
datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.0,
        zoom_range=0.01,
        channel_shift_range=0.0,
        fill_mode='constant',
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        dim_ordering='th') 


def iterate_minibatches(inputs1 , targets,  batchsize, shuffle=False, augment=True):
    assert len(inputs1) == len(targets)
    if augment==True:
        if shuffle:
            indices = np.arange(len(inputs1))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs1) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            x = inputs1[excerpt]
            y = targets[excerpt] 
            for  xxt,yyt in datagen.flow(x, y , batch_size=x.shape[0]):
                x = xxt.astype(np.float32) 
                y = yyt 
                break
    else:
        x=inputs1
        y=targets

    #yield x, np.array(y, dtype=np.uint8)         
    return x, np.array(y, dtype=np.uint8)         


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
    Y=np.array(Y,dtype='uint8')
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



def extract_dsb(ids,augmentation=False,dataset='train'):

    # pre-processing 
    param_prep0={
        'h': H,
        'w': W,
        'crop'    : None,
        'norm_type' : 'minmax_bound',
        'output' : 'mask',
    }

    
    if dataset=='train':
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
    
    for id in ids:    
        X1=f2[id]
        y1=f2[id].attrs['cancer']
        y.append(y1)
        #print ff_dsb_nodes[id].keys()
        
        # obtain nodules 
        Yp_seg=np.array(ff_dsb_nodes[id]['Y'])>0.5
        
        # pixel spacing
        spacing=ff_spacing[id].value

        # find area in mm2
        sumY=np.sum(Yp_seg,axis=(1,2,3))*spacing[0]*spacing[1]*4
        #print sumY
        
        # sort areas
        sY_sorti=np.argsort(-sumY)        
        
        # pick areas within 2mm to 30 mm
        sYp_sort=[]
        for ind in sY_sorti:
            if sumY[ind]>=5 and sumY[ind]<=30:         
                #print sumY[ind]
                sYp_sort.append(ind)
    
        #print sYp_sort
        if len(sYp_sort)==0:
            sYp_sort=np.random.randint(len(Yp_seg),size=10)
                
        top_nodes=z/2
        XY0=[]
        step=3
        for t in sYp_sort[:top_nodes]:   
            X0=X1[t*step+step]
            X0=utils.preprocess(X0[np.newaxis,np.newaxis],param_prep0)

            # nodule mask
            # return to original size
            Y0=cv2.resize(Yp_seg[t,0].astype('uint8'), (W, H), interpolation=cv2.INTER_CUBIC)                
            X0=np.append(X0,Y0[np.newaxis,np.newaxis],axis=1)
            coords=mask2coord(Y0)
            X0,_=crop_nodule_roi(X0,None,coords)
            XY0.append(X0[0])
            
        XY0=np.vstack(XY0)[np.newaxis]
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
    trn_c, val_c, trn_yc, val_yc = cross_validation.train_test_split(cancer,y_c, random_state=420, stratify=y_c,test_size=0.2) 


    # indices of train and validation
    trn_ids=np.concatenate((trn_nc,trn_c))
    trn_y=np.concatenate((trn_ync,trn_yc))

    val_ids=np.concatenate((val_nc,val_c))
    val_y=np.concatenate((val_ync,val_yc))

    # shuffle
    trn_ids,trn_y=utils.unison_shuffled_copies(trn_ids,trn_y)
    val_ids,val_y=utils.unison_shuffled_copies(val_ids,val_y)
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


# training params
params_train={
        'h': hc,
        'w': wc,
        'z': z,
        #'timestep': timestep,
        #'optimizer': 'RMSprop()',
        'learning_rate': 3e-6,
        'optimizer': 'Adam',
        #'loss': 'binary_crossentropy',
        'loss': 'categorical_crossentropy',
        #'loss': 'mean_squared_error',
        'nbepoch': 5000,
        'nb_output': 2,
        'nb_filters': 32,    
        'max_patience': 50    
        }
#model=models.model(params_train)
model=models.model(params_train)
model.summary()

path2weights=weightfolder+"/weights.hdf5"

#%%
print ('training in progress ...')


# pre-processing 
param_prep={
    'h': h,
    'w': w,
    'crop'    : None,
    'norm_type' : 'minmax_bound',
    'output' : 'mask',
}


# checkpoint settings
#checkpoint = ModelCheckpoint(path2weights, monitor='val_loss', verbose=0, save_best_only='True',mode='min')

# load last weights
if pre_train:
    if  os.path.exists(path2weights):
        model.load_weights(path2weights)
        print 'previous weights loaded!'
    else:
        raise IOError('weights not found!')
        

# path to csv file to save scores
path2scorescsv = weightfolder+'/scores.csv'
first_row = 'train,test'
with open(path2scorescsv, 'w+') as f:
    f.write(first_row + '\n')
    
    
# Fit the model
start_time=time.time()
scores_test=[]
scores_train=[]
if params_train['loss']=='dice': 
    best_score = 0
    previous_score = 0
else:
    best_score = 1e6
    previous_score = 1e6
patience = 0

# train on non cancer first
c_nc=1

for epoch in range(params_train['nbepoch']):
    print ('epoch: %s,  Current Learning Rate: %.1e' %(epoch,model.optimizer.lr.get_value()))
    seed = np.random.randint(0, 999999)
    #trn_ids_pick=random.sample(trn_ids,256)
    #trn_ids_pick=trn_ids
    trn_ids_pick1=(trn_ids[trn_y==1])
    trn_ids_pick2=(trn_ids[trn_y==0])
    trn_ids_pick2=random.sample(trn_ids_pick2,len(trn_ids_pick1))
    trn_ids_pick=np.concatenate((trn_ids_pick1,trn_ids_pick2))

    bs2=len(trn_ids_pick)
    for t1 in range(0,len(trn_ids_pick),bs2):
        trn_id_batch=trn_ids_pick[t1:t1+bs2]                
        #print t1
    
        # extract nodule masks 
        X_batch,y_batch=extract_dsb(trn_id_batch,augmentation=False)    
                
        #X_batch,_=iterate_minibatches(X_batch,X_batch,X_batch.shape[0],shuffle=False,augment=True)                
        y_batch = np_utils.to_categorical(np.asarray(y_batch,dtype='uint8'))
        
        
        # fit model to data
        class_weight={0:1,1:1.}
        hist=model.fit(X_batch, np.array(y_batch), nb_epoch=1, batch_size=bs,verbose=0,shuffle=True,class_weight=class_weight)    
        #print hist.history       
        
    # evaluate on test and train data
    if epoch==0:    
        X_test,y_test=extract_dsb(val_ids,False) 
        y_test = np_utils.to_categorical(np.asarray(y_test,dtype='uint8'))
        #X_test=X_test[y_test==c_nc]        
        #y_test=y_test[y_test==c_nc]
    score_test=model.evaluate(X_test,np.array(y_test),verbose=0,batch_size=bs)
    
    #score_train=model.evaluate(X_batch,np.array(y_batch),verbose=0,batch_size=bs)
    score_train=hist.history['loss']
    
    if params_train['loss']=='dice': 
        score_test=score_test[1]   
        score_train=score_train[1]
    
    
    print ('score_train: %s, score_test: %s' %(score_train,score_test))
    scores_test=np.append(scores_test,score_test)
    scores_train=np.append(scores_train,score_train)    

    # check if there i s improvement
    if params_train['loss']=='dice': 
        if (score_test>best_score):
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!! viva, improvement!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
            best_score = score_test
            patience = 0
            model.save_weights(path2weights)            
            
        # learning rate schedule
        if score_test<=previous_score:
            #print "Incrementing Patience."
            patience += 1
    else:
        if (score_test<=best_score):
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!! viva, improvement!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
            best_score = score_test
            patience = 0
            model.save_weights(path2weights)            
            
        # learning rate schedule
        if score_test>previous_score:
            #print "Incrementing Patience."
            patience += 1
        
    
    if patience == params_train['max_patience']:
        params_train['learning_rate'] = params_train['learning_rate']/2
        print ("Upating Current Learning Rate to: ", params_train['learning_rate'])
        model.optimizer.lr.set_value(params_train['learning_rate'])
        print ("Loading the best weights again. best_score: ",best_score)
        model.load_weights(path2weights)
        patience = 0
    
    # save current test score
    previous_score = score_test    
    
    # real time plot
    #plt.plot([e],[score_train],'b.')
    #plt.plot([e],[score_test],'b.')
    #display.clear_output(wait=True)
    #display.display(plt.gcf())
    #sys.stdout.flush()
    
    # store scores into csv file
    with open(path2scorescsv, 'a') as f:
        string = str([score_train,score_test])
        f.write(string + '\n')
       

print ('model was trained!')
elapsed_time=(time.time()-start_time)/60
print ('elapsed time: %d  mins' %elapsed_time)          
#%% plot loss

plt.figure(figsize=(15,10))
plt.plot(scores_test)
plt.plot(scores_train)
plt.title('train-validation progress',fontsize=20)
plt.legend(('test','train'), loc = 'upper right',fontsize=20)
plt.xlabel('epochs',fontsize=20)
plt.ylabel('loss',fontsize=20)
plt.grid(True)
plt.show()

print ('best scores train: %.5f' %(np.min(scores_train)))
print ('best scores test: %.5f' %(np.min(scores_test)))          
          
#%%
# loading best weights from training session
print('-'*30)
print('Loading saved weights...')
print('-'*30)
# load best weights

if os.path.exists(path2weights):
    model.load_weights(path2weights)
    print 'weights loaded!'
else:
    raise IOError

# pre-processing 
param_prep={
    'h': h,
    'w': w,
    'crop'    : None,
    'norm_type' : 'minmax_bound',
    'output' : 'coords',
}

#score_test=model.evaluate(utils.preprocess(X_test,param_prep),y_test,verbose=0)
score_test=model.evaluate(X_test,y_test,verbose=0)
#score_train=model.evaluate(preprocess(X_train,param_prep),y_train,verbose=0)
#print ('score_train: %s, score_test: %s' %(score_train,score_test))
print (' score_test: %s' %(score_test))

print('-'*30)
print('Predicting masks on test data...')
print('-'*30)
#y_pred=model.predict(preprocess(X_test,Y_test,param_prep)[0])
#y_pred=model.predict(preprocess(X_train,Y_train,param_prep)[0])
#%%

tt='test'

# pre-processing 
param_prep={
    'h': h,
    'w': w,
    'crop'    : None,
    'norm_type' : 'minmax_bound',
    'output' : 'mask',
}

if tt is 'train':
    X=X_batch
    y=y_batch
else:
    X_dsb,y_dsb=extract_dsb(val_ids,False)        
    #score_test=model.evaluate(preprocess(np.append(X_test,X,axis=0),param_prep),np.append(y_test,y,axis=0),verbose=0)
    #X=utils.preprocess(X_dsb,param_prep)
    #X=preprocess(X_test,param_prep)
    #X=preprocess(X_dsb,param_prep)
    X=X_dsb    
    y=y_dsb
    #y=y_test
    #y=y_dsb

# prediction
logloss_test=model.evaluate(X_test,y_test,verbose=0)    
y_pred=model.predict(X)
delta=np.abs((y-y_pred[:,0]>0.5)*1.)
plt.plot(y_pred[:,0])

print 'loss test: %s' %logloss_test
print 'accuracy: %.2f' %(1-np.sum(delta)/len(y))
#from sklearn.metrics import log_loss

#y1=np.asanyarray(y_pred>.5,'uint8')
y1 = y_pred[:,0].copy()
print 'logloss: %s' %(utils.logloss(y,y1))


#%%

ff_dsbtest=h5py.File(path2dsbtest_nodes,'r')
print 'total files:', len(ff_dsbtest)


X_dsb_test,_=extract_dsb(ff_dsbtest.keys(),False,'test')
y_pred_test=model.predict(X_dsb_test)    

# sample slice
n1=np.random.randint(y_pred_test.shape[0])
XwithY=utils.image_with_mask(X_dsb_test[n1,0],X_dsb_test[n1,1])
plt.imshow(XwithY)
plt.title(y_pred_test[n1])

# create submission
try:
    df = pd.read_csv('./output/submission/stage1_submission.csv')
    df['cancer'] = y_pred_test[:,0]
except:
    raise IOError    

now = datetime.datetime.now()
info='1slices'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
sub_file = os.path.join('./output/submission', 'submission_' + suffix + '.csv')

df.to_csv(sub_file, index=False)
print(df.head())

#%%

n1=np.random.randint(y_batch.shape[0])

#plt.subplot(1,3,1)
XwithY=utils.image_with_mask(X_batch[n1,0],X_batch[n1,1])
plt.imshow(XwithY)
plt.title([n1,y_batch[n1]])

plt.subplot(1,3,2)
XwithY=utils.image_with_mask(X_batch[n1,2],X_batch[n1,3])
plt.imshow(XwithY)

plt.subplot(1,3,3)
XwithY=utils.image_with_mask(X_batch[n1,4],X_batch[n1,5])
plt.imshow(XwithY)

plt.title(y_batch[n1])
