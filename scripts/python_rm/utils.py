# -*- coding: utf-8 -*-

from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt, cm
from sklearn.externals import joblib
from keras import backend as K
from skimage import measure
#import architectures
import numpy as np
import scipy as sp
#import config
#import data
import sys
import cv2
import os

#%%

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def logloss(act, pred):
    epsilon = 1e-5
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll
    

# preprocess X and Y
def preprocess_XY(X,Y,param_prep):
    # X,Y: n,c,h,w
    N,C,H,W=X.shape
    
    if Y is None:
        Y=np.zeros_like(X,dtype='uint8')
    else:
        N,Cy,H,W=Y.shape        
    
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
        Y_r=np.zeros([N,Cy,h,w],dtype='uint8')
        for k1 in range(N):
            for k2 in range(C):
                X_r[k1,k2,:] = cv2.resize(X[k1,k2], (w, h), interpolation=cv2.INTER_CUBIC)
            for k3 in range(Cy):                
                Y_r[k1,k3,:] = cv2.resize(Y[k1,k3], (w, h), interpolation=cv2.INTER_CUBIC)                
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
    elif norm_type is 'minmax_bound':        
        # normalization
        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0
        
        X_r = (X_r - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        X_r[X_r>1] = 1.
        X_r[X_r<0] = 0.

    # center coordinates and diameter
    #y_r=mask2coord(Y_r)            
    #print X_r.shape
    #print Y_r.shape
    #if output is 'mask':    
    return X_r,Y_r
    #elif output is 'coords':
        #return X_r,y_r


# resize
def array_resize(X,(h,w)):
    X=np.array(X,dtype='uint8')
    N,C,H,W=X.shape
    X_r=np.zeros([N,C,h,w],dtype=X.dtype)
    for k1 in range(N):
        for k2 in range(C):
            X_r[k1,k2,:] = cv2.resize(X[k1,k2], (w, h), interpolation=cv2.INTER_CUBIC)
    return X_r            


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
    return np.array(image,'float32')
    #return image


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
    img_color[mask_edges, 0] = maximg*color[0]  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask_edges, 1] = maximg*color[1]
    img_color[mask_edges, 2] = maximg*color[2]
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
    C2=(0,0,255)
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



# save porition
#def save_portion(X,y):
#    N=100
#    n1=np.random.randint(X_train.shape[0],size=N)
#    X=X[n1]
#    y=y[n1]
#    # save fast train data
#    np.savez(path2output+"fast_dataXy",X=X,y=y)



# convert mask to coordinates
def mask2coord(Y,nb_output=3):
    N,C,H,W=Y.shape
    coords=np.zeros((N,3))
    for k in range(N):
        region=measure.regionprops(Y[k,0])
        if len(region)>0:
            (x,y),radius = cv2.minEnclosingCircle(region[0].coords)
            coords[k,:]=[x,y,radius]
    R=100
    #print np.max(coords[:,2])
    coords=coords/[H,W,R]
    #coords=coords[:,:nb_output]
    return coords

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


#%%
def show_image(image):
    '''
    Plots an image
    
    Args:
        image(numpy): image to be shown
    
    Returns:
        void
    '''
    plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
    
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def create_directory(directory):
    '''
    Creates a directory if it does not exist.
    
    Args:
        directory(str): the location where the directory is to be created
    
    Returns:
        void
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)

def initialize_logger(logger_path):
    '''
    Initialize the Logger. This will ensure we print out the output to the terminal and to a log file.
    
    Args:
        logger_path(str): the location to save the log file
    
    Returns:
        void
    '''
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush() # If you want the output to be visible immediately
        def flush(self) :
            for f in self.files:
                f.flush()
    
    f = open(logger_path, 'w')
    
    original = sys.stdout
    sys.stdout = Tee(original, f)

def save_mask_while_training(X, true_2D_masks, predicted_2D_masks, number_of_rows, number_of_cols, epoch, writing_directory, data_type):
    '''
    This function is meant to save mask results to disk.
    
    Args:
        true_2D_masks (numpy): true (2D) mask
        predicted_2D_masks (numpy): predicted (2D) mask
        number_of_rows(int): number of rows in the figure
        number_of_cols(int): number of columns in the figure
        epoch(int): the name of the figure when saved to disk
        writingDirectory(str): the directory to save the figure
    
    Returns:
        void
    '''
    figure, axes_list = plt.subplots(number_of_rows, number_of_cols, sharex='col', sharey='row')
    
    X = X[:,0,:,:]
    true_2D_masks = true_2D_masks[:,0,:,:]
    predicted_2D_masks = predicted_2D_masks[:,0,:,:]

    index = 0
    for i in range(number_of_rows):
        j = 0
        while j < number_of_cols:
            axes_list[i][j].imshow(X[index],  cmap = cm.Greys_r)
            j += 1
            axes_list[i][j].imshow(true_2D_masks[index],  cmap = cm.Greys_r)
            j += 1
            axes_list[i][j].imshow(predicted_2D_masks[index],  cmap = cm.Greys_r)
            j += 1
            index += 1
        
    for i in range(number_of_rows):
        for j in range(number_of_cols):
            axes_list[i][j].axis('off')
    
    output_directory = writing_directory+'/'+data_type+'_'+'mask_tracking_progress'
    create_directory(output_directory)
    figure.savefig(output_directory+'/'+str(epoch)+'.png',format='png', dpi=300)
    
    # This fixes the memory leak problem of running the script for a long time (large number of epochs)
    for i in range(number_of_rows):
        for j in range(number_of_cols):
            axes_list[i][j].cla()
            
    figure.clf()
    plt.clf()
    plt.close()
    
    del figure
    del axes_list

def get_dice_coefficient_score(X,Y):
    '''
    Computes the dice coefficient.
    
    Args:
        X(numpy array): a 2D true mask
        Y(numpy array): a 2D predicted mask
    
    Returns:
        score: the dice coefficient.
    '''
    
    X = X>0
    Y = Y>0
    
    if np.sum(X) == 0 and np.sum(Y) == 0:
        score=1.0
    else:
        score = (2.0*np.sum(X&Y))/(np.sum(X)+np.sum(Y))

    return score

def compute_score(y_true, y_pred, original_image_height, original_image_width):
    '''
    Computes the score for a set of images.
    
    Args:
        y_true(numpy): 
        y_pred(numpy): 
        original_image_height(int): 
        original_image_width(int): 
    '''
    preds = np.zeros(np.shape(y_true),dtype=np.uint8)
    trues = np.zeros(np.shape(y_true),dtype=np.uint8)
    
    score = 0.0
    for i in range(np.shape(y_pred)[0]):
        true = y_true[i,0]
        if np.sum(true)!=0:
            thresh = threshold_otsu(true)
            true = cv2.threshold(true.astype(np.float32), thresh, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
        trues[i] = true
        true = cv2.resize(true, (original_image_width, original_image_height))
        
        pred = y_pred[i,0]
        if np.sum(pred)!=0:
            thresh = threshold_otsu(pred)
            pred = cv2.threshold(pred.astype(np.float32), thresh, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
        preds[i] = pred
        pred = cv2.resize(pred, (original_image_width, original_image_height))
        
        score += get_dice_coefficient_score(true,pred)
    
    score /= i
    
    return score, preds, trues

def load_data(data_dirs):
    print '****************************'
    print 'Loading: ', data_dirs
    # Load the images data
    for index in range(len(data_dirs)):
        if index == 0:
            images = joblib.load(data_dirs[index])
        else:
            images = np.concatenate((images, joblib.load(data_dirs[index])), axis=1)
        
        #print 'Images Statistics - Channel ', index,':', np.mean(images[:,index,:,:]), np.std(images[:,index,:,:]), np.min(images[:,index,:,:]), np.max(images[:,index,:,:])
    
    #print 'Images Statistics:', np.mean(images), np.std(images), np.min(images), np.max(images)
    
    return images


def standardize_augment_data(images, masks, params, aug_params, seed):
    # Standardize/Augment the validation data
    X,y_masks = data.augment(images, masks, params['pool'], seed, aug_params)
    for index in range(np.shape(X)[1]):
        X[:,index] -= np.mean(images[:,index,:,:])
        std = np.std(images[:,index,:,:])
        if std!=0.0:
            X[:,index] /= std
        
    y_is_positive = None
    if type(y_masks) == np.ndarray:
        y_is_positive = y_masks[:,0,:,:].sum(axis=(1,2))>0
        if params['train_positive_only']:
            X = X[y_is_positive]
            y_masks = y_masks[y_is_positive]
    
        y_is_positive = y_is_positive.astype(np.float32)
    
    return X,y_masks, y_is_positive

def standardize_semi_supervised_data(images, params, aug_params):
    # Standardize/Augment the validation data
    X,_ = data.augment(images, None, params['pool'], aug_params)
    for index in range(np.shape(X)[1]):
        X[:,index] -= np.mean(images[:,index,:,:])
        std = np.std(images[:,index,:,:])
        if std!=0.0:
            X[:,index] /= std
#        print 'X Statistics - Channel ', index,': ', np.mean(X[:,index,:,:]), np.std(X[:,index,:,:]), np.min(X[:,index,:,:]), np.max(X[:,index,:,:])
    
    return X

def get_best_weights_file(writing_directory,weights_type='best'):
    if weights_type=='best':
        model_weights_dir = writing_directory+'/weights.h5'
    else:
        model_weights_dir = writing_directory+'/last_weights.h5'
    
    return model_weights_dir

def convert_mask_to_run_length_encoding(mask):
    '''
    Converts a mask to a run length encoding list.
    
    Args:
        mask: a mask image.
    
    Returns:
        A Run Legnth Encoding List.
    '''

    run_length_encoding = []
    if np.sum(mask>0):
        # Transpose and flatten the mask    
        flattened = mask.T.flatten()
        
        # find the indices of the region of interest
        roi_indices = np.where(flattened>0)[0]
        
        # Append a 0 and a large number at the beginning and end of the roi_indices
        appended_roi_indices = np.concatenate(([0],roi_indices,np.array([(config.original_image_width*config.original_image_height)*10])))
        
        # After that, find where the difference is higher than 1
        large_differences_indices = np.where(np.diff(appended_roi_indices)>1)[0]
        lengths = np.diff(large_differences_indices)
        
        # Get the end and start indices
        end_indices = appended_roi_indices[np.diff(appended_roi_indices)>1][1:]
        start_indices = end_indices-lengths+2
        
        # Combine Results
        run_length_encoding = np.zeros(np.shape(start_indices)[0]*2,dtype=np.int)
        run_length_encoding[::2] = start_indices
        run_length_encoding[1::2] = lengths
        
        run_length_encoding = list(run_length_encoding)
        run_length_encoding = map(str,run_length_encoding)
        
        run_length_encoding = ' '.join(run_length_encoding)
    else:
        run_length_encoding = ''
    
    return run_length_encoding

def convert_run_length_encoding_to_mask(run_length_encoding):
    '''
    Converts a list (numpy array) of run length encoding to a mask
    
    Args:
        run_length_encoding (numpy array): run length encoding representation
    
    Returns:
        mask: a 2D mask
    '''
    mask = np.zeros(config.original_image_height*config.original_image_width)
    
    if run_length_encoding != []:
        # even items are from pixels
        from_pixels = run_length_encoding[::2]-1# in python indices start from 0
        
        # length pixels are odd items
        length_pixels = run_length_encoding[1::2]
        
        # to pixels are the sum of from+length
        to_pixels = from_pixels+length_pixels
        
        # set corresponding pixels to 1
        for index in range(np.shape(from_pixels)[0]):
            mask[from_pixels[index]:to_pixels[index]]=1
    
    # Convert the array to 2D. Note: you need to transpose it
    mask = np.reshape(mask,(config.original_image_width,config.original_image_height)).T
    
    return mask



def get_model(params):
    if params['architecture'] == 'Model1':
        model = architectures.get_Model1(params)
    elif params['architecture'] == 'Model2':
        model = architectures.get_Model2(params)
    elif params['architecture'] == 'Model3':
        model = architectures.get_Model3(params)
    elif params['architecture'] == 'Model4':
        model = architectures.get_Model4(params)
    elif params['architecture'] == 'VGG16':
        model = architectures.get_VGG16(params)
    elif params['architecture'] == 'ResNet':
        model = architectures.get_ResNet(params)
    elif params['architecture'] == 'FCN_old':
        model = architectures.get_FCN_old(params)
    elif params['architecture'] == 'UNET_old':
        model = architectures.get_UNET_old(params)
    elif params['architecture'] == 'FCN_new':
        model = architectures.get_FCN_new(params)
    elif params['architecture'] == 'UNET_new':
        model = architectures.get_UNET_new(params)
    
    elif params['architecture'] == 'FCN_Model1':
        model = architectures.get_FCN_Model1(params)
    elif params['architecture'] == 'FCN_Model2':
        model = architectures.get_FCN_Model2(params)
    elif params['architecture'] == 'FCN_Model3':
        model = architectures.get_FCN_Model3(params)
    elif params['architecture'] == 'FCN_Model4':
        model = architectures.get_FCN_Model4(params)
    elif params['architecture'] == 'FCN_VGG16':
        model = architectures.get_FCN_VGG16(params)
    
    
    else:
        print 'Unsupported Model'
    
    return model

def make_batches(size, batch_size):
    '''Returns a list of batch indices (tuples of indices).
    '''
    nb_batch = int(np.ceil(size / float(batch_size)))
    
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]

def get_intermediate_output_function(model, layer_name):
    '''
    Get the output of any layer in a model
    '''
    return K.function([model.layers[0].input,K.learning_phase()], [model.get_layer(layer_name).output])

def predict_intermediate_output(X, model, layer_name, params):
    '''
    predicts the output of an intermediate layer
    '''
    get_intermediate_output = get_intermediate_output_function(model, layer_name)
    
    intermediate_output = np.zeros((np.shape(X)[0],params['initial_channels'],np.shape(X)[2],np.shape(X)[3]),dtype=np.float32)
    
    for batch_index, (batch_start, batch_end) in enumerate(make_batches(X.shape[0], params['batch_size'])):
        X_batch = X[batch_start:batch_end]
        
        intermediate_batch_output = get_intermediate_output([X_batch,0])[0]
        
        intermediate_output[batch_start:batch_end] = intermediate_batch_output
        
    return intermediate_output
    
    
    