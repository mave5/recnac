#%% nodule segmentation for LUNA dataset

import numpy as np
import cv2
import time
import os
import matplotlib.pylab as plt
#from skimage import measure
import models
import utils
from keras import backend as K
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
import h5py
import random
from glob import glob
from image import ImageDataGenerator

#%%

# path to dataset
path2luna_external="/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/"
path2luna_XY=path2luna_external+'luna_XY.hdf5'
ff_luna_XY=h5py.File(path2luna_XY,'r')
subset_list=ff_luna_XY.keys()
subset_list.sort()
print 'total subsets: %s' %len(subset_list)

#%%

# original data dimension
H = 512
W = 512

# pre-processed data dimesnsion
z,h,w=1,512,512

# image and label channels
c_out=1

# batch size
bs=16

# fold number
foldnm=1

# exeriment name to record weights and scores
experiment='fold'+str(foldnm)+'_luna_lung_seg'+'_hw_'+str(h)+'by'+str(w)+'_cin'+str(z)
print ('experiment:', experiment)

# seed point
seed = 2017
seed = np.random.randint(seed)

# checkpoint
weightfolder='./output/weights/'+experiment
if  not os.path.exists(weightfolder):
    os.makedirs(weightfolder)
    print ('weights folder created')

# data augmentation 
augmentation=True


# fast train
pre_train=True


########## log
import datetime
path2logs='./output/logs/'
now = datetime.datetime.now()
info='log_LunaLungSegmentation_'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
# Direct the output to a log file and to screen
loggerFileName = os.path.join(path2logs,  suffix + '.txt')
utils.initialize_logger(loggerFileName)


#%%

# random data generator
datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.01,
        zoom_range=0.01,
        channel_shift_range=0.0,
        fill_mode='constant',
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


# load train data
def load_data(subsets,display=False):
    X=[]
    Y=[]
    for ss in subsets:
        if display:
            print ss
        for key in ff_luna_XY[ss].keys():
            if display:
                print key
            X0=ff_luna_XY[ss][key]['X']
            Y0=ff_luna_XY[ss][key]['Y']
            rnd_inds=random.sample(xrange(X0.shape[0]), 10)
            rnd_inds.sort()
            if display:            
                print X0.shape,X0.dtype
                print Y0.shape,Y0.dtype
                print rnd_inds
            X.append(X0[rnd_inds])
            Y.append(Y0[rnd_inds])
    X=np.vstack(X)    
    #print X.shape
    Y=np.vstack(Y)
    #print Y.shape
    return X,Y

#%%

print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

# train test split
ss_test=subset_list.pop(foldnm)
ss_train=subset_list
print 'test:', ss_test
print 'train:', ss_train

#X_train,Y_train=load_data(ss_train)        
#utils.array_stats(X_train)
#utils.array_stats(Y_train)
X_test,Y_test=load_data([ss_test])
X_test=utils.normalize(X_test)
# extract lung only
Y_test=(Y_test==3)|(Y_test==4)    

#utils.array_stats(X_test)
#utils.array_stats(Y_test)

    
#%%
print('-'*30)
print('Creating and compiling model...')
print('-'*30)

# training params
params_train={
    'h': h,
    'w': w,
    'z': z,           
    'weights_path': None, 
    'initial_lr': 3e-5,       
    'learning_rate': 1.5e-5,
    'optimizer': 'Adam',
    #'loss': 'binary_crossentropy',
    #'loss': 'mean_squared_error',
    'loss': 'dice',
    'nbepoch': 2000,
    'nb_output': c_out,
    'nb_filters': 16,    
    'max_patience': 50    
        }

model = models.seglung_model(params_train)
#model=models.seg_encode_decode(params_train)
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
    'output' : 'mask',
}

# checkpoint settings
#checkpoint = ModelCheckpoint(path2weights, monitor='val_loss', verbose=0, save_best_only='True',mode='min')

# load last weights
if pre_train:
    if  os.path.exists(path2weights) and pre_train:
        model.load_weights(path2weights)
        print 'weights loaded!'
    else:
        raise IOError

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
    # extract X,Y for training        
    X_train,Y_train=load_data(ss_train)        
    
    # schedule learning rate, start with small value
    #if best_score<0.14:
        #print 'dice is low'
        #K.set_value(model.optimizer.lr,params_train['initial_lr'])
    
    #else:
        #K.set_value(model.optimizer.lr,params_train['learning_rate'])

    print ('epoch: %s,  Current Learning Rate: %.1e' %(epoch,model.optimizer.lr.get_value()))
    seed = np.random.randint(0, 999999)

    # data augmentation
    bs2=64
    for k in range(0,X_train.shape[0],bs2):
        X_batch=X_train[k:k+bs2]
        Y_batch=Y_train[k:k+bs2]
        Y_batch=(Y_batch==3)|(Y_batch==4)
        
        # preprocess 
        #X_batch,Y_batch=utils.preprocess_XY(X_batch,Y_batch,param_prep)
        X_batch=utils.normalize(X_batch)
        
        # augmentation
        #X_batch,Y_batch=iterate_minibatches(X_batch,Y_batch,X_batch.shape[0],shuffle=False,augment=True)        
        hist=model.fit(X_batch[:,np.newaxis], Y_batch[:,np.newaxis,:], nb_epoch=1, batch_size=bs,verbose=0,shuffle=True)
    
    # evaluate on test and train data
    score_test=model.evaluate(X_test[:,np.newaxis],Y_test[:,np.newaxis],verbose=0,batch_size=bs)

    if params_train['loss']=='dice': 
        score_test=score_test[1]   
        score_train=hist.history['dice_coef']
        #score_train=score_train[1]
   
    print ('score_train: %s, score_test: %s' %(score_train,score_test))
    scores_test=np.append(scores_test,score_test)
    scores_train=np.append(scores_train,score_train)    

    # check if there is improvement
    if params_train['loss']=='dice': 
        if (score_test>=best_score):
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!! viva, improvement!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
            best_score = score_test
            patience = 0
            model.save_weights(path2weights)       
            model.save(weightfolder+'/model.h5')
            
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
            model.save(weightfolder+'/model.h5')
            
        # learning rate schedule
        if score_test>previous_score:
            #print "Incrementing Patience."
            patience += 1
    # save anyway    
    #model.save_weights(path2weights)      
            
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
plt.legend(('test','train'), loc = 'lower right',fontsize=20)
plt.xlabel('epochs',fontsize=20)
plt.ylabel('loss',fontsize=20)
plt.grid(True)
plt.show()
plt.savefig(weightfolder+'/train_val_progress.png')

print ('best scores train: %.5f' %(np.max(scores_train)))
print ('best scores test: %.5f' %(np.max(scores_test)))          
          
#%%
# loading best weights from training session
print('-'*30)
print('Loading saved weights...')
print('-'*30)
# load best weights

if  os.path.exists(path2weights):
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

score_test=model.evaluate(*utils.preprocess_XY(X_test,Y_test,param_prep),verbose=0)
score_train=model.evaluate(*utils.preprocess_XY(X_train,Y_train,param_prep),verbose=0)
print ('score_train: %.2f, score_test: %.2f' %(score_train[1],score_test[1]))

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
    X,Y=utils.preprocess_XY(X_train[n1],Y_train[n1],param_prep)
else:
    X,Y=utils.preprocess_XY(X_test,Y_test,param_prep)

# prediction
Y_pred=model.predict(X)>.5    

plt.figure(figsize=(20,20))
n1=utils.disp_img_2masks(np.expand_dims(X[:,3,:],axis=1),np.array(Y_pred,'uint8'),Y,4,4,0)
plt.show()
#%%
# deply on test data
#path2luna='./output/data/luna/test_luna_nodules_cin7.hdf5'
#ff_test=h5py.File(path2luna,'w-')
#
#print ss_test
#t1=h5py.File(ss_test[0],'r')
#for k in t1.keys():
#    print k
#    XY=t1[k]
#    X0=XY[0]
#    Y0=XY[1]
#    X1=[]
#    step=c_in
#    for k2 in range(0,X0.shape[0]-c_in,step):
#        X1.append(X0[k2:k2+c_in])
#    X1=np.stack(X1)
#    Y_pred=model.predict(utils.preprocess(X1,param_prep))>.5
#    ff_test[k]=Y_pred    
#ff_test.close()        


#Y_pred=utils.array_resize(Y_pred,(256,256))
#X0p,Y0p=utils.preprocess_XY(X0[::step,np.newaxis],Y0[::step,np.newaxis],param_prep)
#utils.disp_img_2masks(X1[1],Y_pred[1],None,4,5,0,range(0,20))

#%%     
X_batch,Y_batch=utils.preprocess_XY(X_batch,Y_batch,param_prep)
c1=1
X1=X_batch[:,c1,:]
Y1=Y_batch[:,c1,:]
X1=X1[:,np.newaxis,:]
Y1=Y1[:,np.newaxis,:]
plt.figure()
n1=utils.disp_img_2masks(X1,Y1,None,3,3,0,range(8))
plt.show()
#%%
n1=np.random.randint(X_test.shape[0])
X1=utils.normalize(X_test[n1])
Y1=(Y_test[n1]==3)| (Y_test[n1]==4)
X1=utils.image_with_mask(X1,Y1)
plt.imshow(X1)
