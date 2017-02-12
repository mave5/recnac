# usage: python classify_nodes.py nodes.npy 

import numpy as np
#import pickle

from sklearn import cross_validation
#from sklearn.cross_validation import StratifiedKFold as KFold
#from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF
import xgboost as xgb
from glob import glob
from skimage import measure
from sklearn import cross_validation
import os
import cv2
import datetime
import pandas as pd
import ntpath
import matplotlib.pylab as plt
from skimage.draw import circle
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#%%

# path to dsb data in numpy 
path2dsb_numpy="/media/mra/win7/data/misc/kaggle/datascience2017/notebooks/output/data/dsb/"

path2output='./output/numpy/dsb/'

# original data size
H,W=512,512

# AI model data size
h,w=256,256

# max radius
R=100

#%%
def getRegionFromMap(slice_npy):
    thr = np.where(slice_npy > np.mean(slice_npy),1,0)
    label_image = measure.label(thr)
    labels = label_image.astype(int)
    regions = measure.regionprops(labels)
    return regions

# convert coordinates to mask
def coord2mask(y_pred,params):
    h,w,R=params
    n=len(y_pred)
    Y_pred=np.zeros((n,1,h,w),dtype='uint8')
    for k1 in range(n):
        img = np.zeros((h, w), dtype=np.uint8)
        r,c,radius=y_pred[k1,:]*[h,w,R]
        rr, cc = circle(r,c,radius)
        img[rr, cc] = 1
        Y_pred[k1,:]=img
    return Y_pred    


   
def array_stats(X):
    X=np.asarray(X)
    print ('array shape: ',X.shape, X.dtype)
    #print 'min: %.3f, max:%.3f, avg: %.3f, std:%.3f' %(np.min(X),np.max(X),np.mean(X),np.std(X))
    print ('min: {}, max: {}, avg: {:.3}, std:{:.3}'.format( np.min(X),np.max(X),np.mean(X),np.std(X)))

def normalize(X):
    # normalization
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    X = (X - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    X[X>1] = 1.
    X[X<0] = 0.
    return X

# preprocess
def preprocess(X,Y,param_prep):
    # X,Y: n,c,h,w
    N,C,H,W=X.shape
    
    if Y is None:
        Y=np.zeros_like(X,dtype='uint8')
    
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
        Y_r=np.zeros([N,C,h,w],dtype='uint8')
        for k1 in range(X.shape[0]):
            X_r[k1] = cv2.resize(X[k1,0], (w, h), interpolation=cv2.INTER_CUBIC)
            Y_r[k1] = cv2.resize(Y[k1,0], (w, h), interpolation=cv2.INTER_CUBIC)>0.5 # binary mask 
    else:
        X_r=X
        Y_r=Y

    
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
    y_r=mask2coord(Y_r)            
    
    if output is 'mask':    
        return X_r,Y_r
    elif output is 'coords':
        return X_r,y_r

# pre-processing 
param_prep={
    'h': h,
    'w': w,
    'crop'    : None,
    'norm_type' : 'minmax_bound',
    'output' : 'mask',
}   

# convert mask to coordinates
def mask2coord(Y):
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
    coords=coords[:,:3]
    return coords


def getRegionMetricRow(fnode_name,fimgs_name,params):
    # fname, numpy array of dimension [#slices, 1, 512, 512] containing the images
    f1 = np.load(fnode_name)
    ynode0=f1['Y'] # nodule 
    
    h,w,R=params
    ynode=ynode0*[h,w,R]
    r=ynode[:,0]
    c=ynode[:,1]

    # window
    delta=2
    
    # diff two concecutive rows
    diff2_r=np.abs(np.ediff1d(r))
    
    # sum of distance between two diffs
    r_wcons=[]
    for n in range(len(r)-delta):
        r_wcons.append(np.sum(diff2_r[n:n+delta]))    
    
    # diff two concecutive cols    
    diff2_c=np.abs(np.ediff1d(c))
    
    c_wcons=[]
    for n in range(len(c)-delta):
        c_wcons.append(np.sum(diff2_c[n:n+delta]))    
    
    # convert to array
    r_wcons=np.array(r_wcons)
    c_wcons=np.array(c_wcons)
    
    # find nearest nodules    
    ind_nearest=np.argmin(c_wcons+r_wcons)

    # load images
    #f2 = np.load(fimgs_name)
    #X=f2['X'] # nodule 
    #y=f2['y']
    
    # select nodule slices
    #X=X[ind_nearest:ind_nearest+delta+1]
    
    # reshape to N*C,H*W
    #X=X[np.newaxis,:]
    
    #return X,ynode0[ind_nearest:ind_nearest+delta+1]
    return ind_nearest,delta


def createFeatureDataset(nodfiles):
    print ('wait ...')

    # ground truth data
    #df_train = pd.read_csv('./output/submission/stage1_labels.csv')

    #truth_metric = np.zeros((len(nodfiles)))
    
    inds=[]
    for i,nodfile in enumerate(nodfiles):
        print ('processing: %s, %s' %(i,nodfile))
        
        # extract subject id
        #subject_id=ntpath.basename(nodfile)
        #subject_id=subject_id.replace("_nodule.npz","")
        
        # ground truth label, cancer 0/1
        #truth_metric[i] = int(df_train.cancer[df_train.id == subject_id])
        
        # path to images
        #path2imgs=path2dsb_numpy+subject_id+'.npz'
        
        # get the slices with nodules        
        #X0,y_node= getRegionMetricRow(nodfile,path2imgs,(h,w,R))
        ind_nearest,_= getRegionMetricRow(nodfile,None,(h,w,R))
        #disp_img_2masks(preprocess(X,None,param_prep)[0],None,Y_pred,1,3,0)
        inds.append(ind_nearest)
    #return X,truth_metric
    return inds    

import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

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


#%%

# path to nodule files obtained from segmentation
path2noduls='./output/numpy/dsb/'
nodefile_list=glob(path2noduls+'*.npz')
print ('total nodes files: %s' %(len(nodefile_list)))

# path to csv file to save scores
path2scorescsv = path2output+'/log.csv'
first_row = 'subject number,file name'
with open(path2scorescsv, 'w+') as f:
    f.write(first_row + '\n')

# ground truth data
df_train = pd.read_csv('./output/submission/stage1_labels.csv')


if os.path.exists(path2noduls+"dataY.npy"):
    Y=np.load(path2noduls+"dataY.npy")
    X=np.load(path2noduls+"dataX.npy")
else:    
    # get indices for the nearest nodules
    inds_nears=createFeatureDataset(nodefile_list)

    # ground truth labels init
    y = np.zeros((len(nodefile_list)))    
    
    X=np.zeros((len(y),3,H,W),dtype='int16')
    # loop over node file lists
    for k,nodfile in enumerate(nodefile_list):
        print 'processing %s, %s' %(k,nodfile)
        # extract subject id
        subject_id=ntpath.basename(nodfile)
        subject_id=subject_id.replace("_nodule.npz","")
        
        # ground truth label, cancer 0/1
        y[k] = int(df_train.cancer[df_train.id == subject_id])
        
        # path to images
        path2imgs=path2dsb_numpy+subject_id+'.npz'

        try:
            f1=np.load(path2imgs)
            X0=f1['X']
            X0=X0[inds_nears[k]:inds_nears[k]+3]
            #X.append(X0) 
            X[k,:]=X0
        except:
            # store brocken npz into csv file
            with open(path2scorescsv, 'a') as f:
                string = str([k,nodfile])
                f.write(string + '\n')        
            #continue            
    
   # save features
    np.savez(path2noduls+"dataXy", X=X,y=y,nodefile_list=nodefile_list)
    print 'data saved!'
dasdas    
#%%
trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(X, Y, random_state=420, stratify=Y,
                                                                   test_size=0.1)
#%%

# test data prediction

# path to nodule files obtained from segmentation
path2noduls='./output/numpy/dsb_test/'
nodefile_list=glob(path2noduls+'*.npz')
print ('total nodes files: %s' %(len(nodefile_list)))

if os.path.exists(path2noduls+"dataY.npy"):
    Y=np.load(path2noduls+"dataY.npy")
    X=np.load(path2noduls+"dataX.npy")
else:    
    # create features from the nodule files
    X,Y=createFeatureDataset(nodefile_list)

    # save features
    np.save(path2noduls+"dataY.npy", Y)
    np.save(path2noduls+"dataX.npy", X)

Y_test = clf.predict(X)


# create submission
try:
    df = pd.read_csv('./output/submission/stage1_submission.csv')
    df['cancer'] = Y_test
except:
    raise IOError    

now = datetime.datetime.now()
info='threestages'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
sub_file = os.path.join('./output/submission', 'submission_' + suffix + '.csv')

df.to_csv(sub_file, index=False)
print(df.head())

