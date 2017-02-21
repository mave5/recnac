 #from __future__import print_function

import numpy as np
#from keras.models import Model
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from keras import backend as K
#from sklearn.externals import joblib
from sklearn import cross_validation
#import random
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
bs=32

# trained data dimesnsion
h,w=512,512

# exeriment name to record weights and scores
experiment='dsb_final_classify_aug'+'_hw_'+str(h)+'by'+str(w)
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
pre_train=True

# top n
top_n=1
z=top_n*3
#%%

# functions
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
#from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
#from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten,Dense
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
    model.add(Convolution2D(C, 3, 3, activation='relu',subsample=(4,4),border_mode='same', input_shape=(z, h, w)))

    N=6
    for k in range(1,N):
        C1=np.min([2**k*C,512])
        model.add(Convolution2D(C1, 3, 3, activation='relu', subsample=(1,1), border_mode='same'))              
        model.add(Convolution2D(C1, 3, 3, subsample=(1,1), activation='relu', border_mode='same'))              
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
    model.add(Convolution2D(C1, 3, 3, activation='relu',subsample=(1,1), border_mode='same'))              
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
    
    C1=(0,255,0)
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
#def save_portion(X,y):
#    N=100
#    n1=np.random.randint(X_train.shape[0],size=N)
#    X=X[n1]
#    y=y[n1]
#    # save fast train data
#    np.savez(path2output+"fast_dataXy",X=X,y=y)

# extract three consecutive images from BS subject
def extract_dsb(trn_nc,bs,df_train,top_n):
    
    # read h5 file
    f2=h5py.File(path2dsb,'r')
    f3=h5py.File(weightfolder+'/dsb_pred.hdf5','r')
    
    # initialize   
    X=np.zeros((bs,z,H,W),'int16')
    y=np.zeros(bs,dtype='uint8')
    for k,ind in enumerate(trn_nc):
        p_id=df_train.id[ind]
        # label        
        y[k]=df_train.cancer[ind]
        #print k,ind,p_id,y[k]
        
        # images 
        X0=f2[p_id]
        
        # predictions
        y0=f3[p_id]
        
        # pick top n predictions        
        step=3
        yp=np.array(y0[::step,0])
        #print yp
        ind_topn = (-yp).argsort()[:top_n]
        ind_topn.sort()
        #print ind_topn
        #print np.ediff1d(ind_topn)
        #print ind_topn
    
        # pick top suspicious images
        Xk=[]
        for t1 in ind_topn:  
            #print ind
            Xk.append(X0[step*t1:step*t1+3])
            #print 3*t1,3*t1+3
        Xk=np.vstack(Xk)        
        X[k,:]=Xk
        #print y[k]
    f2.close()    
    f3.close()    
    return X,y


def logloss(act, pred):
    epsilon = 1e-5
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

from itertools import islice, chain

def batch(iterable, size):
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)
        yield chain([batchiter.next()], batchiter)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
    
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

########## load DSB data, only non-cancer
df_train = pd.read_csv(path2csv)
print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
df_train.head()

# extract non cancers
non_cancer=df_train[df_train.cancer==0].index
cancer=df_train[df_train.cancer==1].index
print 'total non cancer:%s, total cancer:%s' %(len(non_cancer),len(cancer))

y_nc=np.zeros(len(non_cancer),'uint8')
y_c=np.ones(len(cancer),'uint8')

# train val split
#path2trainvalindx=weightfolder+'/dsb_train_val_index'
trn_nc, val_nc, trn_ync, val_ync = cross_validation.train_test_split(non_cancer,y_nc, random_state=420, stratify=y_nc,test_size=0.1)                                                                   
trn_c, val_c, trn_yc, val_yc = cross_validation.train_test_split(cancer,y_c, random_state=420, stratify=y_c,test_size=0.1) 

# indices of train and validation
trn_ind=np.concatenate((trn_nc,trn_c))
trn_y=np.concatenate((trn_ync,trn_yc))

val_ind=np.concatenate((val_nc,val_c))
val_y=np.concatenate((val_ync,val_yc))

# shuffle
trn_ind,trn_y=unison_shuffled_copies(trn_ind,trn_y)
val_ind,val_y=unison_shuffled_copies(val_ind,val_y)

# test data
X_test,y_test=extract_dsb(val_ind,len(val_ind),df_train,top_n)        
    
#%%
print('-'*30)
print('Creating and compiling model...')
print('-'*30)

# training params
params_train={
    'img_rows': h,
    'img_cols': w,
    'img_depth':z,
    'weights_path': None,        
    'learning_rate': 3e-5,
    'optimizer': 'Adam',
    'loss': 'binary_crossentropy',
    #'loss': 'mean_squared_error',
    #'loss': 'dice',
    'nbepoch': 1000,
    'nb_output': nb_output,
    'nb_filters': 16,    
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

    #for batchiter in batch(trn_ind, bs):
    for t1 in range(0,len(trn_ind),bs):
        trn_ind_batch=trn_ind[t1:t1+bs]                
        #print trn_ind_batch
        # extract a random batch from DSB
        X_batch,y_batch=extract_dsb(trn_ind_batch,bs,df_train,top_n)    
        
        # data preprocssing and augmentation    
        X_batch=preprocess(X_batch,param_prep)
        X_batch,y_batch=iterate_minibatches(X_batch,y_batch,X_batch.shape[0],shuffle=False,augment=True)

        # fit model to data
        model.fit(X_batch, y_batch, nb_epoch=1, batch_size=bs,verbose=0,shuffle=True)    
        #print model.evaluate(X_batch,y_batch,verbose=0)
    
    # evaluate on test and train data
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


