from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.externals import joblib
from image import ImageDataGenerator
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import time
import os
import matplotlib.pylab as plt
from skimage import measure, morphology, segmentation
#%%
working_path = "./output/numpy/luna/allsubsets/"

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.

# batch size
bs=16

h,w=512,512
experiment='coords'+'_hw_'+str(h)+'by'+str(w)
print ('experiment:', experiment)

seed = 2016
np.random.seed=seed

# checkpoint
weightfolder='./output/weights/'+experiment
if  not os.path.exists(weightfolder):
    os.makedirs(weightfolder)
    print ('weights folder created')

# data augmentation
augmentation=True

#%%

# functions
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D,Dropout,AtrousConvolution2D
from keras.layers import Activation,Reshape,Permute,Flatten,Dense
from keras.layers.advanced_activations import ELU
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.models import Sequential
#from funcs.image import ImageDataGenerator

def dice_coef(y_true, y_pred):
    smooth=1
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
    return K.mean(intersection / union)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# model
def model(params):

    h=params['img_rows']
    w=params['img_cols']
    lr=params['learning_rate']
    weights_path=params['weights_path']
    loss=params['loss']
    C=params['nb_filters']
    
    model = Sequential()
    
    model.add(Convolution2D(C, 3, 3, activation='relu',subsample=(2,2),border_mode='same', input_shape=(1, h, w)))

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

    model.add(Dense(3, activation='sigmoid'))
    
    #load previous weights
    if weights_path:
        model.load_weights(weights_path)

    model.compile(loss=loss, optimizer=Adam(lr))

    return model
    
# preprocess
def preprocess(X,Y,param_prep):
    # X,Y: n,c,h,w
    N,C,H,W=X.shape
    
    # get params
    h=param_prep['h']
    w=param_prep['w']    
    crop=param_prep['crop']
    norm_type=param_prep['norm_type'] # normalization 
    
    
    # center crop h*w
    if crop is 'center':
        hc=(H-h)/2
        wc=(W-w)/2
        X=X[:,:,hc:H-hc,wc:W-wc]
        Y=Y[:,:,hc:H-hc,wc:W-wc]
    elif crop is 'random':
        hc=(H-h)/2
        wc=(W-w)/2
        hcr=np.random.randint(hc)
        wcr=np.random.randint(wc)
        X=X[:,:,hcr:H-hcr,wcr:W-wcr]
        Y=Y[:,:,hcr:H-hcr,wcr:W-wcr]
        
    # check if need to downsample
    # resize if needed
    if h<H:
        X_r=np.zeros([N,C,h,w],dtype=X.dtype)
        Y_r=np.zeros([N,C,h,w],dtype='uint8')
        for k1 in range(X.shape[0]):
            X_r[k1] = cv2.resize(X[k1,0], (w, h), interpolation=cv2.INTER_CUBIC)
            Y_r[k1] = (cv2.resize(Y[k1,0], (w, h), interpolation=cv2.INTER_CUBIC)>0.5)
    else:
        X_r=X
        Y_r=Y
    
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
            
    return X_r,Y_r

# calcualte dice
def calc_dice(X,Y,d=0):
    N=X.shape[d]    
    # intialize dice vector
    dice=np.zeros([N,1])

    for k in range(N):
        x=X[k,0] >.5 # convert to logical
        y =Y[k,0]>.5 # convert to logical

        # number of ones for intersection and union
        intersectXY=np.sum((x&y==1))
        unionXY=np.sum(x)+np.sum(y)

        if unionXY!=0:
            dice[k]=2* intersectXY/(unionXY*1.0)
            #print 'dice is: %0.2f' %dice[k]
        else:
            dice[k]=1
            #print 'dice is: %0.2f' % dice[k]
        #print 'processing %d, dice= %0.2f' %(k,dice[k])
    return np.mean(dice),dice
    


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


MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

# convert mask to coordinates
def mask2coord(Y):
    N=Y.shape[0]
    coords=np.zeros((N,3))
    for k in range(N):
        region=measure.regionprops(Y[k,0])
        if len(region)>0:
            (x,y),radius = cv2.minEnclosingCircle(region[0].coords)
            coords[k,:]=[x,y,radius]
    return coords


# random data generator
datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.01,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        dim_ordering='th') 

def iterate_minibatches(inputs1 , targets,  batchsize, shuffle=True, augment=True):
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
X_train = joblib.load(working_path+"trainX.joblib").astype('float32')
X_train=normalize(X_train)
Y_train = joblib.load(working_path+"trainY.joblib")
array_stats(X_train)
array_stats(Y_train)

# porition
N=480
n1=np.random.randint(X_train.shape[0],size=N)
X_train=X_train[n1]
Y_train=Y_train[n1]

X_test = joblib.load(working_path+"testX.joblib").astype('float32')
X_test=normalize(X_test)
Y_test = joblib.load(working_path+"testY.joblib")
array_stats(X_test)
array_stats(Y_test)

N=50
n1=np.random.randint(X_test.shape[0],size=N)
X_test=X_test[n1]
Y_test=Y_test[n1]


# pre-processing 
param_prep={
    'h': h,
    'w': w,
    'crop'    : None,
    'norm_type' : 'local',
}

#X_train,Y_train=preprocess(X_train,Y_train,param_prep)
#array_stats(X_train)
#array_stats(Y_train)
#n1=disp_img_2masks(X_train,Y_train,None,3,4,True)

X_test,Y_test=preprocess(X_test,Y_test,param_prep)
#array_stats(X_test)
#array_stats(Y_test)

#n1=disp_img_2masks(X_test,Y_test,None,4,5,True)
#%%
# convert masks to coordinates
#y_train=mask2coord(Y_train)/(h*1.)
y_test=mask2coord(Y_test)/(h*1.)
        
#%%
print('-'*30)
print('Creating and compiling model...')
print('-'*30)

# training params
params_train={
    'img_rows': h,
    'img_cols': w,           
    'weights_path': None,        
    'learning_rate': 3e-4,
    'optimizer': 'Adam',
    #'loss': 'binary_crossentropy',
    'loss': 'mean_squared_error',
    #'loss': 'dice',
    'nbepoch': 1000,
    'num_labels': 3,
    'nb_filters': 8,    
    'max_patience': 50    
        }

model = model(params_train)
model.summary()

# path to weights
path2weights=weightfolder+"/weights.hdf5"

#%%
          
print ('training in progress ...')

# checkpoint settings
checkpoint = ModelCheckpoint(path2weights, monitor='val_loss', verbose=0, save_best_only='True',mode='min')

# load last weights
#if  os.path.exists(path2weights):
#    model.load_weights(path2weights)

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

    # data augmentation
    if augmentation is True:
        X_train_aug,Y_train_aug=iterate_minibatches(X_train,Y_train , X_train.shape[0], shuffle=True,augment=True)
    else:
        X_train_aug,Y_train_aug=X_train,Y_train

    # normalize        
    X_train_aug,Y_train_aug=preprocess(X_train_aug,Y_train_aug,param_prep)
        
    # convert masks to coordinates
    y_train_aug=mask2coord(Y_train_aug)/(h*1.)

    # fit model
    model.fit(X_train_aug, y_train_aug, validation_data=(X_test, y_test), nb_epoch=1, batch_size=bs,verbose=0,shuffle=True,callbacks=[checkpoint])
    
    # evaluate on test and train data
    score_test=model.evaluate(X_test, y_test,verbose=0)
    score_train=model.evaluate(X_train_aug, y_train_aug,verbose=0)
    if params_train['loss']=='dice': 
        score_test=score_test[1]   
        score_train=score_train[1]
   
    print ('score_train: %s, score_test: %s' %(score_train,score_test))
    scores_test=np.append(scores_test,score_test)
    scores_train=np.append(scores_train,score_train)    

    # check if there is improvement
    if params_train['loss']=='dice': 
        if (score_test>best_score):
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!! viva, improvement!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
            best_score = score_test
            patience = 0
        # learning rate schedule
        if score_test<=previous_score:
            #print "Incrementing Patience."
            patience += 1
    else:
        if (score_test<=best_score):
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!! viva, improvement!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
            best_score = score_test
            patience = 0
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

scores=pd.read_csv(path2scorescsv)
strain=scores.train
stest=scores.test
scores_train=[float(s[1:]) for s in strain]
scores_test=[float(s[:-1]) for s in stest]    

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
model.load_weights(path2weights)


print('-'*30)
print('Predicting masks on test data...')
print('-'*30)


# normalize        
X_train_aug,Y_train_aug=preprocess(X_train,Y_train,param_prep)
y_train_aug=mask2coord(Y_train_aug)/(h*1.)
train_loss=model.evaluate(X_train_aug,y_train_aug)
test_loss=model.evaluate(X_test,y_test)
print ('score_train: %s, score_test: %s' %(train_loss,test_loss))

y_pred=model.predict(X_test)
#y_pred=model.predict(X_train_aug)

#%%
from skimage.draw import circle

y=(y_pred*h)#.astype('int16')
X=X_test
Y=Y_test

Y_pred=np.zeros_like(Y)

for k1 in range(y.shape[0]):
    img = np.zeros((h, w), dtype=np.uint8)
    r,c,radius=y[k1,:]
    rr, cc = circle(r,c,  radius)
    img[rr, cc] = 1
    Y_pred[k1,:]=img

Y1=Y
Y2=Y_pred
plt.figure(figsize=(20,20))
#plt.figure()
n1=disp_img_2masks(X,Y1,Y2,4,4,0)
plt.show()

