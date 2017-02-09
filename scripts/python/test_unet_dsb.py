from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import cv2
import time
from glob import glob
import os
import matplotlib.pylab as plt
from sklearn.externals import joblib
from skimage import measure
from skimage.transform import resize
import ntpath
#%%
#working_path = "./output/numpy/dsb/"
#working_path ="/media/mra/win7/data/misc/kaggle/datascience2017/notebooks/output/data/dsb/"
working_path ="/media/mra/win7/data/misc/kaggle/datascience2017/notebooks/output/data/dsb_test/"

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
path2output='./output/numpy/dsb_test/'

img_rows = 512
img_cols = 512

smooth = 1.

h,w=128,128
experiment='unet'+'_hw_'+str(h)+'by'+str(w)
print ('experiment:', experiment)

seed = 2017
np.random.seed=seed

# checkpoint
weightfolder='./output/weights/'+experiment
if  not os.path.exists(weightfolder):
    os.makedirs(weightfolder)
    print ('weights folder created')

#%%

# functions
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D,Dropout,AtrousConvolution2D
from keras.layers import Activation,Reshape,Permute
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
    lossfunc=params['loss']
    C=params['nb_filters']
    
    model = Sequential()
    
    model.add(Convolution2D(C, 3, 3, activation='relu',border_mode='same', input_shape=(1, h, w)))

    N=5
    for k in range(1,N):
        model.add(Convolution2D(2**k*C, 3, 3, activation='relu', border_mode='same'))              
        model.add(Convolution2D(2**k*C, 3, 3, subsample=(2,2), activation='relu', border_mode='same'))              
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        
    model.add(Convolution2D(2**k*C, 3, 3, activation='relu', border_mode='same'))              
    model.add(Convolution2D(2**k*C, 3, 3, activation='relu', border_mode='same'))              
    model.add(Dropout(0.5))

    for k in range(1,N):
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(2**(N-k)*C, 3, 3, activation='relu', border_mode='same'))
    
    model.add(Convolution2D(C, 3, 3, activation='relu', border_mode='same'))              
    model.add(Convolution2D(1, 1, 1, activation='sigmoid', border_mode='same'))                            
    

    #load previous weights
    if weights_path:
        model.load_weights(weights_path)

    if lossfunc=='dice':
        model.compile(optimizer=Adam(lr), loss=dice_coef_loss, metrics=[dice_coef])
    else:
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr))

    return model

# preprocess
def preprocess(X,Y,param_prep):
    # X,Y: n,c,h,w
    N,C,H,W=X.shape
    
    if Y is None:
        Y=np.zeros_like(X)
    
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
        X=X[:,:,hc:H-hcr,wc:W-wcr]
        Y=Y[:,:,hc:H-hcr,wc:W-wcr]
        
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
    

def preprocess0(imgs_to_process,masks):
    # imgs_to_process: N*H*W

    out_images = []      #final set of images
    out_masks = []   #final set of nodemasks

    for i in range(len(imgs_to_process)):
        mask = masks[i]
        #node_mask = node_masks[i]
        img = imgs_to_process[i]
        new_size = [512,512]   # we're scaling back up to the original size of the image
        img= mask*img          # apply lung mask
        #
        # renormalizing the masked image (in the mask region)
        #
        new_mean = np.mean(img[mask>0])  
        new_std = np.std(img[mask>0])
        #
        #  Pulling the background color up to the lower end
        #  of the pixel range for the lungs
        #
        old_min = np.min(img)       # background color
        img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
        img = img-new_mean
        img = img/new_std
        #make image bounding box  (min row, min col, max row, max col)
        labels = measure.label(mask)
        regions = measure.regionprops(labels)
        #
        # Finding the global min and max row over all regions
        #
        min_row = 512
        max_row = 0
        min_col = 512
        max_col = 0
        for prop in regions:
            B = prop.bbox
            if min_row > B[0]:
                min_row = B[0]
            if min_col > B[1]:
                min_col = B[1]
            if max_row < B[2]:
                max_row = B[2]
            if max_col < B[3]:
                max_col = B[3]
        width = max_col-min_col
        height = max_row - min_row
        if width > height:
            max_row=min_row+width
        else:
            max_col = min_col+height
        # 
        # cropping the image down to the bounding box for all regions
        # (there's probably an skimage command that can do this in one line)
        # 
        img = img[min_row:max_row,min_col:max_col]
        mask =  mask[min_row:max_row,min_col:max_col]
        if max_row-min_row <5 or max_col-min_col<5:  # skipping all images with no god regions
            pass
        else:
            # moving range to -1 to 1 to accomodate the resize function
            mean = np.mean(img)
            img = img - mean
            min = np.min(img)
            max = np.max(img)
            img = img/(max-min)
            new_img = resize(img,[512,512])
            new_mask = resize(255*mask,[512,512]).astype(np.uint8)
            out_images.append(new_img)
            out_masks.append(new_mask)
    out_images=np.asanyarray(out_images,'float32')            
    out_images=out_images[:,np.newaxis,:]
    #out_masks=np.asanyarray(out_masks,'uint8')            
    return out_images

def array_stats(*args):
    for X in args:
        X=np.asarray(X)
        print ('array shape: ',X.shape, X.dtype)
        #print 'min: %.3f, max:%.3f, avg: %.3f, std:%.3f' %(np.min(X),np.max(X),np.mean(X),np.std(X))
        print ('min: {}, max: {}, avg: {:.3}, std:{:.3}'.format( np.min(X),np.max(X),np.mean(X),np.std(X)))

#%%

print('-'*30)
print('Creating and compiling model...')
print('-'*30)

# training params
params_train={
    'img_rows': h,
    'img_cols': w,           
    'weights_path': None,        
    'learning_rate': 3e-5,
    'optimizer': 'Adam',
    'loss': 'binary_crossentropy',
    'nbepoch': 300,
    'num_labels': 2,
    'nb_filters': 8,    
    'max_patience': 20    
        }

model = model(params_train)
model.summary()

# path to weights
path2weights=weightfolder+"/weights.hdf5"

if not os.path.isfile(path2weights):
    raise IOError("Path to weights does not exist!")
else:
    # load best weights
    model.load_weights(path2weights)

#%%
print('-'*30)
print ('wait ...')

# pre-processing 
param_prep={
    'h': h,
    'w': w,
    'crop'    : None,
    'norm_type' : None,
}

# path to csv file to save scores
path2scorescsv = path2output+'/log.csv'
first_row = 'subject number,file name'
with open(path2scorescsv, 'w+') as f:
    f.write(first_row + '\n')

# get the list of lung masks
masks_list=glob(working_path+"*_lungmask.joblib")

subject_num=0
for subject_num,fname in enumerate(masks_list):
    start_time=time.time()
    
    #subject_num+=1    
    print ("processing subject %s: %s " %(subject_num, fname))
    try:
        f1 = np.load(fname.replace("_lungmask.joblib",".npz"))
        imgs_test=f1['X']
        cancer=f1['y']
        masks=joblib.load(fname)
    except:
        # store brocken npz into csv file
        with open(path2scorescsv, 'a') as f:
            string = str([subject_num,fname])
            f.write(string + '\n')        
        continue            
        

    # first step of pre-processing
    imgs_test=preprocess0(imgs_test,masks)

    # second step pre-processing
    X_test,Y_test=preprocess(imgs_test,None,param_prep)
    #array_stats(X_test,Y_test)
          
    # prediction      
    Y_pred=model.predict(X_test)>0.5
    
    # pick nonzero masks    
    nz_slices=np.sum(Y_pred,axis=(1,2,3))
    Y_pred_nz=Y_pred[nz_slices>0]

    # extract subject id
    subject_id=ntpath.basename(fname)
    subject_id=subject_id.replace("_lungmask.joblib","_nodule")
    
    # save nodules
    np.savez(path2output+subject_id,Y=Y_pred_nz,y=cancer,nzY=nz_slices)
    
    elapsed_time=(time.time()-start_time)
    print ('elapsed time: %.1f  sec' %(elapsed_time))
    print ('-'*30)
#%%

X=X_test#[ysum>0]
Y1=Y_pred#[ysum>0]

plt.figure(figsize=(20,100))
n1=disp_img_2masks(X,Y1,None,4,4,0,range(48,64))

