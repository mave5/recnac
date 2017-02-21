 #from __future__import print_function

import numpy as np
#from keras.models import Model
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from keras import backend as K
#from sklearn.externals import joblib
from sklearn import cross_validation
import random
import datetime
import utils
#from image import ImageDataGenerator
#from augmentation import CustomImageDataGenerator
#import augmentation
#from augmentation import random_zoom, elastic_transform 
from keras.preprocessing.image import ImageDataGenerator
import cv2
import time
import os
import matplotlib.pylab as plt
import scipy as sp
import h5py    
#from skimage import measure, morphology, segmentation
#%%

root_data='/media/mra/win7/data/misc/kaggle/datascience2017/notebooks'
path2dsb=root_data+'/output/data/dsb.hdf5'
#path2dsb="/media/mra/My Passport/Kaggle/datascience2017/dsb/"
path2luna=root_data+'/output/data/luna/'
path2csv=root_data+'/output/data/stage1_labels.csv'
path2logs='./output/logs/'

# dispaly
display_ena=False

# resize
H,W=512,512

#K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

# original data dimension
img_rows = 512
img_cols = 512

# batch size
bs=16

# trained data dimesnsion
h,w=256,256

# exeriment name to record weights and scores
experiment='ClassifyLUNAvsDSB_aug'+'_hw_'+str(h)+'by'+str(w)
print ('experiment:', experiment)

# seed point
seed = 2016
seed = np.random.randint(seed)

# checkpoint
weightfolder='./output/weights/'+experiment
if  not os.path.exists(weightfolder):
    os.makedirs(weightfolder)
    print ('weights folder created')

# data augmentation
augmentation=True

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
pre_train=False

#%%

# functions
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D,Dropout
from keras.optimizers import Adam
#from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint ,LearningRateScheduler
#from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Activation,Reshape,Permute,Flatten,Dense
#from keras.layers.advanced_activations import ELU
#from keras.models import Model
from keras import backend as K
#from keras.optimizers import Adam#, SGD
from keras.models import Sequential
#from funcs.image import ImageDataGenerator


# model
def model(params):

    h=params['img_rows']
    w=params['img_cols']
    z=params['img_depth']
    lr=params['learning_rate']
    weights_path=params['weights_path']
    loss=params['loss']
    C=params['nb_filters']
    nb_output=params['nb_output']
    
    model = Sequential()
    model.add(Convolution2D(C, 3, 3, activation='relu',subsample=(2,2),border_mode='same', input_shape=(z, h, w)))

    N=6
    for k in range(1,N):
        C1=np.min([2**k*C,512])
        model.add(Convolution2D(C1, 3, 3, activation='relu', subsample=(1,1), border_mode='same'))              
        model.add(Convolution2D(C1, 3, 3, subsample=(1,1), activation='relu', border_mode='same'))              
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
    model.add(Convolution2D(C1, 3, 3, activation='relu', border_mode='same'))              
    model.add(Convolution2D(C1, 3, 3, activation='relu', border_mode='same'))              
    #model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    #model.add(Dropout(0.1))

    model.add(Dense(nb_output, activation='sigmoid'))
    
    #load previous weights
    if weights_path:
        model.load_weights(weights_path)

    model.compile(loss=loss, optimizer=Adam(lr))

    return model
    
# preprocess
def preprocess(X,param_prep):
    # X,Y: n,c,h,w
    N,C,H,W=X.shape
    
    # get params
    h=param_prep['h']
    w=param_prep['w']    
    crop=param_prep['crop']
    norm_type=param_prep['norm_type'] # normalization 
    output=param_prep['output'] # output
    
    
    # center crop h*w
    if crop is 'center':
        hc=(H-h)/2
        wc=(W-w)/2
        X=X[:,:,hc:H-hc,wc:W-wc]
    elif crop is 'random':
        hc=(H-h)/2
        wc=(W-w)/2
        hcr=np.random.randint(hc)
        wcr=np.random.randint(wc)
        X=X[:,:,hcr:H-hcr,wcr:W-wcr]
        
    # check if need to downsample
    # resize if needed
    if h<H:
        X_r=np.zeros([N,C,h,w],dtype=X.dtype)
        for k1 in range(N):
            for k2 in range(C):
                X_r[k1,k2,:] = cv2.resize(X[k1,k2], (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        X_r=X

    # normalization
    X_r=np.array(X_r,dtype='float32')
    if norm_type is 'global':
        X_r-=np.mean(X_r)
        X_r/=np.std(X_r)
    elif norm_type is 'local':
        for k in range(X_r.shape[0]):
            mean = np.mean(X_r[k,0])  # mean       
            sigma = np.std(X_r[k,0])  # std
            if sigma<1e-5:
                sigma=1
            X_r[k] = X_r[k]-mean
            X_r[k] = X_r[k]/ sigma
    elif norm_type is 'scale':
        X_r-=np.min(X_r)
        X_r/=np.max(X_r)
    elif norm_type is 'minmax_bound':        
        # normalization
        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0
        
        X_r = (X_r - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        X_r[X_r>1] = 1.
        X_r[X_r<0] = 0.

    return X_r


####### normalization
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def grays_to_RGB(img):
    # turn 2D grayscale image into grayscale RGB
    return np.dstack((img, img, img))


def image_with_mask(img, mask,color=(0,255,0)):
    maximg=np.max(img)    
    img=np.asarray(img,dtype='float32')
    img=np.asarray((img/maximg)*255,dtype='uint8')
    mask=np.asarray(mask,dtype='uint8') 
    if np.max(mask)==1:
        mask=mask*255

    # returns a copy of the image with edges of the mask added in red
    if len(img.shape)==2:	
	img_color = grays_to_RGB(img)
    else:
	img_color =img

    mask_edges = cv2.Canny(mask, 100, 200) > 0
    img_color[mask_edges, 0] = color[0]  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask_edges, 1] = color[1]
    img_color[mask_edges, 2] = color[2]
    img_color=img_color#/float(np.max(img))
    return img_color

def disp_img_2masks(img,mask1,mask2,r=1,c=1,d=0,indices=None):
    if mask1 is None:
        mask1=np.zeros(img.shape,dtype='uint8')
    if mask2 is None:
        mask2=np.zeros(img.shape,dtype='uint8')
        
    N=r*c    
    if d==2:
        # convert to N*C*H*W
        img=np.transpose(img,(2,0,1))
        img=np.expand_dims(img,axis=1)
        
        mask1=np.transpose(mask1,(2,0,1))
        mask1=np.expand_dims(mask1,axis=1)

        mask2=np.transpose(mask2,(2,0,1))
        mask2=np.expand_dims(mask2,axis=1)
        
    if indices is None:    
        # random indices   
        n1=np.random.randint(img.shape[0],size=N)
    else:
        n1=indices
    
    I1=img[n1,0]
    #M1=mask1[n1,0]
    M1=np.zeros(I1.shape,dtype='uint8')
    for c1 in range(mask1.shape[1]):
        M1=np.logical_or(M1,mask1[n1,c1,:])    
    #M2=mask2[n1,0]
    M2=np.zeros(I1.shape,dtype='uint8')
    for c1 in range(mask2.shape[1]):
        M2=np.logical_or(M2,mask2[n1,c1,:])    
    
    C1=(0,255,9)
    C2=(255,0,0)
    for k in range(N):    
        imgmask=image_with_mask(I1[k],M1[k],C1)
        imgmask=image_with_mask(imgmask,M2[k],C2)
        plt.subplot(r,c,k+1)
        plt.imshow(imgmask)
        plt.title(n1[k])
    plt.show()            
    return n1        

def array_stats(X):
    X=np.asarray(X)
    print ('array shape: ',X.shape, X.dtype)
    #print 'min: %.3f, max:%.3f, avg: %.3f, std:%.3f' %(np.min(X),np.max(X),np.mean(X),np.std(X))
    print ('min: {}, max: {}, avg: {:.3}, std:{:.3}'.format( np.min(X),np.max(X),np.mean(X),np.std(X)))


# random data generator
datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.0,
        zoom_range=0.0,
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
            #print "shuffled!"
        for start_idx in range(0, len(inputs1) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
                #print "shuffled", excerpt
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
                
            x = inputs1[excerpt]
            y = targets[excerpt] 
            for  xxt,yyt in datagen.flow(x, y , batch_size=x.shape[0],shuffle=shuffle):
                x = xxt.astype(np.float32) 
                y = yyt 
                break
    else:
        x=inputs1
        y=targets

    #yield x, np.array(y, dtype=np.uint8)         
    return x, np.array(y, dtype=np.uint8)

# save porition
def save_portion(X,y):
    N=100
    n1=np.random.randint(X_train.shape[0],size=N)
    X=X[n1]
    y=y[n1]
    # save fast train data
    np.savez(path2output+"fast_dataXy",X=X,y=y)

# extract three consecutive images from BS subject
def extract_dsb(trn_nc,bs,df_train):
    # pick bs random subjects
    rnd_nc_inds=random.sample(trn_nc,bs)    
    
    # read h5 file
    f2=h5py.File(path2dsb,'r')
    
    # initialize    
    X=np.zeros((bs,3,H,W),'int16')
    y=np.zeros(bs,dtype='uint8')
    for k,ind in enumerate(rnd_nc_inds):
        p_id=df_train.id[ind]
        #print p_id
    
        X0=f2[p_id]
    
        # pick three concecutive slices
        n1=np.random.randint(len(X0)-3)
        #print n1
        X0=X0[n1:n1+3]
        #array_stats(X0) 
        X[k,:]=X0
    
    f2.close()    
    return X,y


def logloss(act, pred):
    epsilon = 1e-5
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

# convert to h5
# create h5 file

#f1=h5py.File(path2luna+'luna.hdf5','w-')
#f1['X_train']=X_train    
#f1['X_test']=X_test
#f1['y_train']=y_train    
#f1['y_test']=y_test
#f1.close()

#%%

print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

### load luna data, all cancer
f1=h5py.File(path2luna+'luna.hdf5','r')
X_train=f1['X_train']
y_train=f1['y_train']
X_test=f1['X_test']
y_test=f1['y_test']

array_stats(X_train)
array_stats(y_train)


########## load DSB data, only non-cancer
df_train = pd.read_csv(path2csv)
print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
df_train.head()

# extract non cancers
non_cancer=df_train[df_train.cancer==0].index
cancer=df_train[df_train.cancer==1].index
print 'total non cancer:%s, total cancer:%s' %(len(non_cancer),len(cancer))

# total non cancers
nb_noncancer=len(non_cancer)
y_dsb=np.zeros(nb_noncancer,dtype='uint8')

# train val split
path2trainvalindx=weightfolder+'/dsb_train_val_index'
if not os.path.exists(path2trainvalindx+'.npz'):
    trn_nc, val_nc, trn_y, val_y = cross_validation.train_test_split(non_cancer,y_dsb, random_state=420, stratify=y_dsb,test_size=0.1)                                                                   
    np.savez(path2trainvalindx,trn_nc=trn_nc,trn_y=trn_y,val_nc=val_nc,val_y=val_y)
else:
    out1=np.load(path2trainvalindx+'.npz')
    trn_nc=out1['trn_nc']
    trn_y=out1['trn_y']    
    val_nc=out1['val_nc']    
    val_y=out1['val_y']    
    print 'previous indices loaded!'
    
#%%
print('-'*30)
print('Creating and compiling model...')
print('-'*30)

# training params
params_train={
    'img_rows': h,
    'img_cols': w,
    'img_depth':3,
    'weights_path': None,        
    'learning_rate': 3e-4,
    'optimizer': 'Adam',
    'loss': 'binary_crossentropy',
    #'loss': 'mean_squared_error',
    #'loss': 'dice',
    'nbepoch': 1000,
    'nb_output': nb_output,
    'nb_filters': 8,    
    'max_patience': 30    
        }

model = model(params_train)
model.summary()

# path to weights
path2weights=weightfolder+"/weights.hdf5"

#%%
          
print ('training in progress ...')

# pre-processing 
param_prep={
    'h': h,
    'w': w,
    'crop'    : None,
    'norm_type' : 'minmax_bound',
    'output' : 'coords',
}

# checkpoint settings
checkpoint = ModelCheckpoint(path2weights, monitor='val_loss', verbose=0, save_best_only='True',mode='min')

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


for e in range(params_train['nbepoch']):
    print ('epoch: %s,  Current Learning Rate: %s' %(e,params_train['learning_rate']))
    seed = np.random.randint(0, 999999)

    
    for k in range(0,X_train.shape[0],bs):
    #for k in range(0,48,bs):
        # extract a batch from luna data
        X_batch=X_train[k:k+bs]
        y_batch=y_train[k:k+bs]

        # extract a random batch from DSB
        X,y=extract_dsb(trn_nc,bs,df_train)        
        X_batch=np.append(X_batch,X,axis=0)
        y_batch=np.append(y_batch,y,axis=0)
    
        # data augmentation    
        X_batch=preprocess(X_batch,param_prep)
        X_batch,y_batch=iterate_minibatches(X_batch,y_batch,X_batch.shape[0],shuffle=False,augment=True)
        # preprocess
        #X_batch=preprocess(X_batch,param_prep)
        model.fit(X_batch, y_batch, nb_epoch=1, batch_size=bs,verbose=0,shuffle=True)    

    
    # evaluate on test and train data
    #X,y=extract_dsb(val_nc,6*bs,df_train)        
    #score_test=model.evaluate(preprocess(np.append(X_test,X,axis=0),param_prep),np.append(y_test,y,axis=0),verbose=0)
    score_test=model.evaluate(preprocess(X_test,param_prep),y_test,verbose=0)
    score_train=model.evaluate(X_batch,y_batch,verbose=0)
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
#%%

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

score_test=model.evaluate(preprocess(X_test,param_prep),y_test,verbose=0)
score_train=model.evaluate(preprocess(X_train,param_prep),y_train,verbose=0)
print ('score_train: %s, score_test: %s' %(score_train,score_test))

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
    n1=np.random.randint(len(X_train),size=100)
    X=preprocess(X_train[n1],param_prep)
    y=y_train[n1]
else:
    X_dsb,y_dsb=extract_dsb(val_nc,6*bs,df_train)        
    #score_test=model.evaluate(preprocess(np.append(X_test,X,axis=0),param_prep),np.append(y_test,y,axis=0),verbose=0)
    X=preprocess(np.append(X_test,X_dsb,axis=0),param_prep)
    #X=preprocess(X_test,param_prep)
    #X=preprocess(X_dsb,param_prep)
    y=np.append(y_test,y_dsb)
    #y=y_test
    #y=y_dsb

# prediction
logloss_test=model.evaluate(X,y,verbose=0)    
y_pred=model.predict(X)
delta=np.abs((y-y_pred[:,0]>0.5)*1.)
plt.plot(y_pred[:,0])

print 'loss test: %s' %logloss_test
print 'accuracy: %.2f' %(1-np.sum(delta)/len(y))
#from sklearn.metrics import log_loss

#y1=np.asanyarray(y_pred>.5,'uint8')
y1 = y_pred[:,0]
print 'logloss: %s' %(logloss(y,y1))


#%%

# non-cancer data

# load data
y_pred=[]
for p in val_nc:
    p_id=df_train.id[p]
    print 'processing: %s %s' %(p,p_id)
    f3=h5py.File(path2dsb,'r')
    X0=f3[p_id]

    X3=[]
    for k in range(X0.shape[0]-3):
        tmp=X0[k:k+3]
        tmp=tmp[np.newaxis,:]
        X3.append(tmp)
    X3=np.vstack(X3)        
    y_p=model.predict(preprocess(X3,param_prep))
    y_pred.append(y_p[:,0])   

# get avg max, avg of top max
yp_avg=[]
yp_max=[]
yp_avgtop=[]
for k in range(len(y_pred)):
    yp_avg.append(np.mean(y_pred[k]))
    yp_max.append(np.max(y_pred[k]))
    n=5
    yk=np.array(y_pred[k])
    idx = (-yk).argsort()[:n]
    yp_avgtop.append(np.mean(yk[idx]))


yp_avg=np.array(yp_avg)
yp_max=np.array(yp_max)


r,c=3,3
for k1 in range(r*c):
    ind=np.random.randint(len(y_pred))
    plt.subplot(r,c,k1+1)
    plt.stem(y_pred[ind])
    plt.show()

# save non-cancer results
np.savez(weightfolder+'/ypred_val_nc',y=y_pred,ids=df_train.id,index=val_nc)

# load
f1=np.load(weightfolder+'/ypred_val_nc.npz')
y_pred_nc=f1['y']
ids=f1['ids']
indices=f1['index']

y_nc=np.zeros_like(yp_max)
print 'avg: %s' %logloss(y_nc,yp_avg)
print 'top avg: %s' %logloss(y_nc,yp_avgtop)
print 'max: %s ' %logloss(y_nc,yp_max)


#%%
# cancer data
# load data
y_pred_cancer=[]
for p in cancer:
    p_id=df_train.id[p]
    print 'processing: %s, %s' %(p,p_id)
    f4=h5py.File(path2dsb,'r')
    X0=f4[p_id]

    X3=[]
    for k in range(X0.shape[0]-3):
        tmp=X0[k:k+3]
        tmp=tmp[np.newaxis,:]
        X3.append(tmp)
    X3=np.vstack(X3)        
    y_p=model.predict(preprocess(X3,param_prep))
    y_pred_cancer.append(y_p[:,0])   


# average, max and avg of top max
yp_cancer_avg=[]
yp_cancer_max=[]
yp_cancer_avgtop=[]
for k in range(len(y_pred_cancer)):
    maxyp=np.max(y_pred_cancer[k])
    yp_cancer_avg.append(np.mean(y_pred_cancer[k])/maxyp)
    yp_cancer_max.append(np.max(y_pred_cancer[k]))
    n=5
    yk=np.array(y_pred_cancer[k])
    idx = (-yk).argsort()[:n]
    yp_cancer_avgtop.append(np.mean(yk[idx]))


# save cancer results
np.savez(weightfolder+'/ypred_cancer',y=y_pred_cancer,ids=df_train.id,index=cancer)

# load
f3=np.load(weightfolder+'/ypred_cancer.npz')
y_pred_cancer=f3['y']
ids=f3['ids']
indices=f3['index']

for k1 in range(16):
    ind=np.random.randint(len(y_pred_cancer))
    plt.subplot(4,4,k1+1)
    plt.stem(y_pred_cancer[ind]/np.max(y_pred_cancer[ind]))
    plt.show()

# log loss
y_c=np.ones_like(yp_cancer_max)
yp_cancer_max=np.array(yp_cancer_max)
print 'max:',logloss(y_c,yp_cancer_max)
print 'avg:',logloss(y_c,yp_cancer_avg)
print 'avg top:',logloss(y_c,yp_cancer_avgtop)


#%%
## verify data augmentation
k1=3
plt.subplot(2,3,1)
plt.imshow(X_batch1[k1,0],cmap='gray')
plt.subplot(2,3,2)
plt.imshow(X_batch1[k1,1],cmap='gray')
plt.subplot(2,3,3)
plt.imshow(X_batch1[k1,2],cmap='gray')
plt.title('augmented')
plt.show()


#k1=10
plt.subplot(2,3,4)
plt.imshow(X_batch[k1,0],cmap='gray')
plt.subplot(2,3,5)
plt.imshow(X_batch[k1,1],cmap='gray')
plt.subplot(2,3,6)
plt.imshow(X_batch[k1,2],cmap='gray')
plt.title('original')
plt.show()
