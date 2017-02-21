import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from sklearn.externals import joblib
from sklearn import cross_validation
import random
import datetime
import utils
import models
from image import ImageDataGenerator

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
path2dsb_nodes='./output/data/dsb/dsb_nodes.hdf5'
#path2luna=root_data+'/output/data/luna/'
path2luna='./output/data/luna/'
path2csv=root_data+'/output/data/stage1_labels.csv'
path2logs='./output/logs/'

# path to nodules
path2output='./output/numpy/dsb/'

# dispaly
display_ena=False



# original data dimension
H,W=512,512

# nodule ROI
hc,wc=128,128

# batch size
bs=32

# trained data dimesnsion
h,w=128,128



# exeriment name to record weights and scores
experiment='classify_nodules'+'_hw_'+str(h)+'by'+str(w)
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


# extract three consecutive images from BS subject
def extract_dsb_seq(trn_nc,bs,df_train):
    # pick bs random subjects
    rnd_nc_inds=random.sample(trn_nc,bs)    
    #print rnd_nc_inds
    
    # path to dsb images
    f2=h5py.File(path2dsb,'r')
    
    # read dsb nodules
    f6=h5py.File(path2dsb_nodes,'r')
    
    # initialize    
    X=[]#np.zeros((0,3,hc,wc),'int16')
    y=[]#np.zeros(0,dtype='uint8')
    for k,ind in enumerate(rnd_nc_inds):
        p_id=df_train.id[ind]
        #print p_id
    
        X0=np.array(f2[p_id],'int16')
        y0node=np.array(f6[p_id],'float32')
        
        X0,_,_=crop_nodule_roi(X0[:,np.newaxis,:],None,None,y0node,reshape_ena=False)
        #utils.array_stats(X0) 
        X.append(X0)
    X=np.vstack(X)
    #utils.array_stats(X)
    n=int(X.shape[0]/3)*3
    #print n
    X=np.reshape(X[:n],(n/3,3,hc,wc))    
    y=np.zeros(len(X),dtype='uint8')
    f2.close()    
    f6.close()    
    return X,y
   

# extract three consecutive images from BS subject
def extract_dsb(trn_nc,bs,df_train):
    # pick bs random subjects
    rnd_nc_inds=random.sample(trn_nc,bs)    
    print rnd_nc_inds
    
    # path to dsb images
    f2=h5py.File(path2dsb,'r')
    
    # read dsb nodules
    f6=h5py.File(path2dsb_nodes,'r')
    
    # initialize    
    X=np.zeros((bs,3,hc,wc),'int16')
    y=np.zeros(bs,dtype='uint8')
    for k,ind in enumerate(rnd_nc_inds):
        p_id=df_train.id[ind]
        #print p_id
    
        X0=np.array(f2[p_id],'int16')
        y0node=np.array(f6[p_id],'float32')
        
        # pick three concecutive slices
        n1=np.random.randint(len(X0)-3)
        #print n1
        X0=X0[n1:n1+3]
        y0node=y0node[n1:n1+3]
        #print X0.shape        
        #print y0node.shape

        X0,_,_=crop_nodule_roi(X0[:,np.newaxis,:],None,None,y0node)
        #print X0.shape
        #array_stats(X0) 
        X[k,:]=X0
    #print X.shape
    f2.close()    
    f6.close()    
    return X,y


# crop ROI
def crop_nodule_roi(X,Y1,Y2,y,reshape_ena=True):

    if Y1 is None:
        Y1=np.zeros_like(X,dtype='uint8')
    if Y2 is None:        
        Y2=np.zeros_like(X,dtype='uint8')

    N,C0,H0,W0=X.shape
    hc,wc=128,128
    
    Xc=np.zeros((N,C0,hc,wc),dtype='int16')
    Yc1=np.zeros((N,C0,hc,wc),dtype='uint8')
    Yc2=np.zeros((N,C0,hc,wc),dtype='uint8')
    for k in range(N):
        r,c,_=y[k,:]*[H0,W0,100]
        #print r,c
        if r>=hc/2 and (r+hc/2)<=H0:
            r1=int(r-hc/2)
            r2=int(r+hc/2)        
        elif r<hc/2:
            r1=0
            r2=int(r1+hc)
        elif (r+hc/2)>H:
            r2=H0
            r1=r2-hc
            

        if c>=wc/2 and (c+wc/2)<=W0:
            c1=int(c-wc/2)
            c2=int(c+wc/2)        
        elif c<wc/2:
            c1=0
            c2=int(c1+wc)
        elif (c+wc/2)>W:
            c2=W0
            c1=c2-wc
            
            
        #print k,c2-c1,r2-r1,r2,c2,X.shape
        Xc[k,:]=X[k,:,r1:r2,c1:c2]
        Yc1[k,:]=Y1[k,:,r1:r2,c1:c2]
        Yc2[k,:]=Y2[k,:,r1:r2,c1:c2]

    # select non zero-masks
    #Xc=Xc[y.sum(axis=1)>0]        
    #Yc1=Yc1[y.sum(axis=1)>0]        
    #Yc2=Yc2[y.sum(axis=1)>0] 

    # reshape to N*3*H*W
    if reshape_ena:
        n,c,hc,wc=Xc.shape
        Xc=np.reshape(Xc,(n/3,3,hc,wc))
        Yc1=np.reshape(Yc1,(n/3,3,hc,wc))
        Yc2=np.reshape(Yc2,(n/3,3,hc,wc))
       
    return Xc,Yc1,Yc2        


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

def logloss(act, pred):
    epsilon = 1e-5
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


# pre-processing 
param_prep={
    'h': h,
    'w': w,
    'crop'    : None,
    'norm_type' : 'minmax_bound',
    'output' : 'coords',
}

#%%

print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

### load luna data, all cancer
f1=h5py.File(path2luna+'luna.hdf5','r')

X_train=f1['X_train']
Y_train=f1['Y_train']
X_test=f1['X_test']
Y_test=f1['Y_test']

# get coordinates from the masks
y_coords_train=utils.mask2coord(Y_train)
y_coors_test=utils.mask2coord(Y_test)

# crop nodule region
X_train,Y_train,_=crop_nodule_roi(X_train,Y_train,None,y_coords_train)
X_test,Y_test,_=crop_nodule_roi(X_test,Y_test,None,y_coors_test)

# set labels for LUNA to ones
y_train=np.ones(len(X_train),dtype='uint8')
y_test=np.ones(len(X_test),dtype='uint8')

utils.array_stats(X_train)
utils.array_stats(Y_train)
utils.array_stats(X_test)
utils.array_stats(Y_test)

plt.figure(figsize=(10,10))
n1=utils.disp_img_2masks(utils.preprocess(X_train,param_prep),Y_train,None,4,3,0)
plt.show()
plt.figure(figsize=(10,10))
n1=utils.disp_img_2masks(utils.preprocess(X_test,param_prep),Y_test,None,4,3,0,None)
plt.show()


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

model = models.model(params_train)
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
    print ('epoch: %s,  Current Learning Rate: %s' %(epoch,params_train['learning_rate']))
    seed = np.random.randint(0, 999999)

    #for k in range(0,X_train.shape[0],bs1):
        #print k
    #for k in range(0,48,bs):
        # extract a batch from luna data
        #X_batch=X_train[k:k+bs]
        #y_batch=y_train[k:k+bs]

    # extract a random batch from DSB
    X_batch,y_batch=extract_dsb_seq(trn_nc,bs,df_train)        
    X_batch=np.append(X_batch,X_train,axis=0)
    y_batch=np.append(y_batch,y_train,axis=0)
    
    # data augmentation    
    X_batch=utils.preprocess(X_batch,param_prep)
    X_batch,y_batch=iterate_minibatches(X_batch,y_batch,X_batch.shape[0],shuffle=False,augment=True)
    model.fit(X_batch, y_batch, nb_epoch=1, batch_size=bs,verbose=0,shuffle=True)    

    
    # evaluate on test and train data
    X,y=extract_dsb_seq(val_nc,bs/2,df_train)        
    score_test=model.evaluate(utils.preprocess(np.append(X_test,X,axis=0),param_prep),np.append(y_test,y,axis=0),verbose=0)
    #score_test=model.evaluate(preprocess(X_test,param_prep),y_test,verbose=0)
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

score_test=model.evaluate(utils.preprocess(X_test,param_prep),y_test,verbose=0)
score_train=model.evaluate(utils.preprocess(X_train,param_prep),y_train,verbose=0)
print ('score_train: %s, score_test: %s' %(score_train,score_test))

print('-'*30)
print('Predicting masks on test data...')
print('-'*30)
#y_pred=model.predict(utils.preprocess(X_test,param_prep))
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
    X_dsb,y_dsb=extract_dsb_seq(val_nc,bs,df_train)        
    #score_test=model.evaluate(preprocess(np.append(X_test,X,axis=0),param_prep),np.append(y_test,y,axis=0),verbose=0)
    X=utils.preprocess(np.append(X_test,X_dsb,axis=0),param_prep)
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
#y1 = y_pred[:,0]
#print 'logloss: %s' %(logloss(y,y1))


#%%

# non-cancer data

# load data
y_pred=[]
for p in val_nc:
    p_id=df_train.id[p]
    print 'processing: %s %s' %(p,p_id)
    f3=h5py.File(path2dsb,'r')
    f31=h5py.File(path2dsb_nodes,'r') # node files
    X0=np.array(f3[p_id])
    y0node=np.array(f31[p_id])

    # crop node roi
    X0,_,_=crop_nodule_roi(X0[:,np.newaxis,:],None,None,y0node,False)

    X3=[]
    step=3
    for k in range(0,X0.shape[0]-3,step):
        tmp=X0[k:k+3,0]
        tmp=tmp[np.newaxis,:]
        X3.append(tmp)
    X3=np.vstack(X3)        
    y_p=model.predict(utils.preprocess(X3,param_prep))
    y_pred.append(y_p[:,0])   

# get avg max, avg of top max
yp_avg=[]
yp_max=[]
yp_avgtop=[]
for k in range(len(y_pred)):
    yp_avg.append(np.mean(y_pred[k]))
    yp_max.append(np.max(y_pred[k]))
    n=3
    yk=np.array(y_pred[k])
    idx = (-yk).argsort()[:n]
    yp_avgtop.append(np.mean(yk[idx]))


yp_avg=np.array(yp_avg)
yp_max=np.array(yp_max)


r,c=3,3
for k1 in range(r*c):
    ind=np.random.randint(len(y_pred))
    plt.subplot(r,c,k1+1)
    #plt.stem(y_pred[ind])
    plt.hist(y_pred[ind])
    plt.show()

# save non-cancer results
#np.savez(weightfolder+'/ypred_val_nc',y=y_pred,ids=df_train.id,index=val_nc)

# load
#f1=np.load(weightfolder+'/ypred_val_nc.npz')
#y_pred_nc=f1['y']
#ids=f1['ids']
#indices=f1['index']

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
    f41=h5py.File(path2dsb_nodes,'r') # node files
    X0=np.array(f4[p_id],'int16')
    
    #y0node=np.array(f41[p_id])
    
    # crop node roi
    #X0,_,_=crop_nodule_roi(X0[:,np.newaxis,:],None,None,y0node,False)
    
    #X3=[]
    #step=1
    #for k in range(0,X0.shape[0]-3,step):
       # tmp=X0[k:k+3,0]
      #  tmp=tmp[np.newaxis,:]
     #   X3.append(tmp)
    #X3=np.vstack(X3).astype('int16')        
    X3,_=extract_dsb_seq([p],1,df_train)            
    y_p=model.predict(utils.preprocess(X3,param_prep))
    print np.max(y_p)
    y_pred_cancer.append(y_p[:,0])   


# average, max and avg of top max
yp_cancer_avg=[]
yp_cancer_max=[]
yp_cancer_avgtop=[]
for k in range(len(y_pred_cancer)):
    #maxyp=np.max(y_pred_cancer[k])
    yp_cancer_avg.append(np.mean(y_pred_cancer[k]))
    yp_cancer_max.append(np.max(y_pred_cancer[k]))
    n=3
    yk=np.array(y_pred_cancer[k])
    idx = (-yk).argsort()[:n]
    yp_cancer_avgtop.append(np.mean(yk[idx]))

# save cancer results
#np.savez(weightfolder+'/ypred_cancer',y=y_pred_cancer,ids=df_train.id,index=cancer)

# load
#f3=np.load(weightfolder+'/ypred_cancer.npz')
#y_pred_cancer=f3['y']
#ids=f3['ids']
#indices=f3['index']

for k1 in range(16):
    ind=np.random.randint(len(y_pred_cancer))
    plt.subplot(4,4,k1+1)
    plt.stem(y_pred_cancer[ind])
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
