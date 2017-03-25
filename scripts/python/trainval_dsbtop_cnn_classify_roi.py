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

# non zero nodes cropped
pathdsb_nz_roi=root_data+'dsb_nz_roi.hdf5'

pathdsb_byid_roi=root_data+'dsb_byid_roi.hdf5'
pathdsb_byid_roi_top=root_data+'dsb_byid_roi_top.hdf5'

path2csv=root_data+'stage1_labels.csv'
path2logs='./output/logs/'

# path to nodes
path2dsbnoduls=root_data+'nfolds_dsb_nodes.hdf5'
path2dsbtest_nodes=root_data+'nfolds_dsbtest_nodes.hdf5'

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
experiment='dsbtop_cnn_classify_'+'roi_hw_'+str(h)+'by'+str(w)+'_cin'+str(c_in)+'_z'+str(z)
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



#%%

print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

########## load DSB data, only non-cancer
df_train = pd.read_csv(path2csv)
print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
df_train.head()


# reald dsb data with top confidence values
ff_dsbbyid_roi_top=h5py.File(pathdsb_byid_roi_top,'r')
print 'total:', len(ff_dsbbyid_roi_top)

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
    trn_c, val_c, trn_yc, val_yc = cross_validation.train_test_split(cancer,y_c, random_state=420, stratify=y_c,test_size=0.1) 

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


# train data
X_train=[]
for id1 in trn_ids:
    X0=ff_dsbbyid_roi_top[id1]
    X_train.append(X0)
X_train=np.stack(X_train)
y_train=trn_y

# test data
X_test=[]
for id1 in val_ids:
    X0=ff_dsbbyid_roi_top[id1]
    X_test.append(X0)
X_test=np.stack(X_test)
y_test=val_y


utils.array_stats(X_train)
utils.array_stats(y_train)

utils.array_stats(X_test)
utils.array_stats(y_test)

#%%

path2segweights=weightfolder+"/weights_seg.hdf5"

# training params
params_train={
        'h': hc,
        'w': wc,
        'z': z,
        'c_in': z,
        'learning_rate': 3e-4,
        'optimizer': 'Adam',
        'loss': 'binary_crossentropy',
        'path2segweights': path2segweights,
        #'loss': 'categorical_crossentropy',
        #'loss': 'mean_squared_error',
        'nbepoch': 5000,
        'nb_output': 1,
        'nb_filters': 32,    
        'max_patience': 50    
        }
#model=models.model(params_train)
#model=models.model(params_train)
model=models.model_pretrain(params_train)        
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


for epoch in range(params_train['nbepoch']):
    print ('epoch: %s,  Current Learning Rate: %.1e' %(epoch,model.optimizer.lr.get_value()))
    seed = np.random.randint(0, 999999)
    
    #hist=model.fit(X_batch, np.array(y_batch), nb_epoch=1, batch_size=bs,verbose=0,shuffle=True,class_weight=class_weight)    
    X_batch,_=iterate_minibatches(X_train,X_train,X_train.shape[0],shuffle=False,augment=True)                
    hist=model.fit(X_batch, y_train, nb_epoch=1, batch_size=bs,verbose=0,shuffle=True)            
    #print hist.history       
        
    # evaluate on test and train data
    score_test=model.evaluate(X_test,y_test,verbose=0,batch_size=bs)
    
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

list2=np.random.choice(X_test1.shape[0], X_test.shape[0], replace=False)    
list2=list(list2)
list2.sort()

score_test=model.evaluate(np.append(X_test,X_test1[list2],axis=0),np.append(y_test,y_test1[list2],axis=0),verbose=0,batch_size=bs)
print (' score_test: %s' %(score_test))
print('-'*30)
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
    X=np.append(X_train,X_train1[list1],axis=0)
    y=np.append(y_train,y_train1[list1],axis=0)
else:
    X=np.append(X_test,X_test1[list2],axis=0)
    y=np.append(y_test,y_test1[list2],axis=0)


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

Xc=ff_dsb_nz_roi['X_c']
yc_pred=model.predict(Xc)
plt.plot(yc_pred[:210,0])
#%%
pathdsb_byid_roi=root_data+'dsb_byid_roi.hdf5'
pathdsb_byid_roi_top=root_data+'dsb_byid_roi_top.hdf5'

ff_dsbbyid_roi=h5py.File(pathdsb_byid_roi,'r')                
ff_dsbbyid_roi_top=h5py.File(pathdsb_byid_roi_top,'w-')                

print ff_dsbbyid_roi.keys()

for id1 in df_train.id:
    print id1
    X_dsb_roi=ff_dsbbyid_roi[id1]
    tmp=model.predict(X_dsb_roi)    
    max_ind=np.argmax(tmp)            
    print 'max prob: ', np.max(tmp),max_ind
    tmp=np.max(tmp)
    ff_dsbbyid_roi_top[id1]=X_dsb_roi[tmp]
    X_dsb_roi[tmp].shape
ff_dsbbyid_roi_top.close()    

# verify
ff_dsbbyid_roi_top=h5py.File(pathdsb_byid_roi_top,'r')                
print 'total:', len(ff_dsbbyid_roi_top)

#%%

# test data
df_test = pd.read_csv('./output/submission/stage1_submission.csv')

pathdsbtest_roi=root_data+'dsbtest_roi.hdf5'
ff_dsbtest_roi=h5py.File(pathdsbtest_roi,'r')
print 'total files:', len(ff_dsbtest_roi)


y_pred_test=[]
for id1 in df_test.id:
    print id1
    X_dsb_test=ff_dsbtest_roi[id1]
    tmp=model.predict(X_dsb_test)    
    max_ind=np.argmax(tmp)            
    print 'max prob: ', np.max(tmp),max_ind
    tmp=np.max(tmp)
    y_pred_test.append(tmp)
    

plt.imshow(X_dsb_test[max_ind,1],cmap='gray')
plt.title(tmp)


# create submission
try:
    df = pd.read_csv('./output/submission/stage1_submission.csv')
    df['cancer'] = y_pred_testn1=np.random.randint(198)
except:
    raise IOError    

now = datetime.datetime.now()
info='node_classify'
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
