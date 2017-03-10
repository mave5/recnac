import numpy as np
import cv2
import time
import os
import matplotlib.pylab as plt
#from skimage import measure
import models
import utils
from sklearn.model_selection import train_test_split
import h5py
from glob import glob
from image import ImageDataGenerator

#%%

path2luna="/media/mra/win7/data/misc/kaggle/datascience2017/LUNA2016/"
subset_list=glob(path2luna+'subset*.hdf5')
print 'total subsets: %s' %len(subset_list)

#%%


# original data dimension
img_rows = 512
img_cols = 512

# pre-processed data dimesnsion
h,w=256,256

# image and label channels
c_in,c_out=7,1


# batch size
bs=32

# exeriment name to record weights and scores
experiment='luna_nodule_seg_'+'_hw_'+str(h)+'by'+str(w)+'_cin'+str(c_in)
print ('experiment:', experiment)

# seed point
seed = 201
seed = np.random.randint(seed)

# checkpoint
weightfolder='./output/weights/'+experiment
if  not os.path.exists(weightfolder):
    os.makedirs(weightfolder)
    print ('weights folder created')

# data augmentation 
augmentation=True


# fast train
pre_train=False

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
def load_data(subsets):
    X=[]
    Y=[]
    for ss in subsets:
        print ss
        ff=h5py.File(ss,'r')
        for k in ff.keys():
            print k
            XY=ff[k]
            nz_Y=np.where(np.sum(XY[1],axis=(1,2))>0)[0]
            nz_Y=nz_Y[1::3]
            #print nz_Y
            step=int(c_in/2)
            X0=[]
            Y0=[]
            for z in nz_Y:
                #print XY.shape[1]
                if z+step+1<XY.shape[1]:
                    XY1=XY[:,z-step:z+step+1]
                else:
                    XY1=XY[:,z-2*step:]
                X0.append(XY1[0])
                Y0.append(XY1[1][step])
            X0=np.stack(X0)    
            Y0=np.stack(Y0)
            Y0=Y0[:,np.newaxis,:]
            #print X0.shape
            #print Y0.shape
            X.append(X0)
            Y.append(Y0)
            
        ff.close()    
    X=np.vstack(X)        
    Y=np.vstack(Y).astype('uint8')        
    return X,Y

#%%

print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

# split to train and test
ss_train, ss_test, _, _ = train_test_split(subset_list, subset_list, test_size=0.1, random_state=42)
np.savez(weightfolder+'/train_test_indx.npz',train=ss_train,test=ss_test)

path2luna_train_test=path2luna+'train_test.hdf5'
if os.path.exists(path2luna_train_test):
    ff_r=h5py.File(path2luna_train_test,'r')
    X_train=ff_r['X_train']
    Y_train=ff_r['Y_train']
    X_test=ff_r['X_test']
    Y_test=ff_r['Y_test']
    print 'hdf5 loaded '
else:    
    # load images and masks    
    X_train,Y_train=load_data(ss_train)   
    X_test,Y_test=load_data(ss_test)   
    ff_w=h5py.File(path2luna_train_test,'w-')
    ff_w['X_train']=X_train
    ff_w['Y_train']=Y_train
    ff_w['X_test']=X_test
    ff_w['Y_test']=Y_test
    ff_w.close()
    print 'hdf5 saved!'

#Y_train=Y_train[:,3,:][:,np.newaxis,:]
#Y_test=Y_test[:,3,:][:,np.newaxis,:]
#%%
print('-'*30)
print('Creating and compiling model...')
print('-'*30)

# training params
params_train={
    'h': h,
    'w': w,
    'c_in': c_in,           
    'weights_path': None,        
    'learning_rate': 3e-5,
    'optimizer': 'Adam',
    #'loss': 'binary_crossentropy',
    #'loss': 'mean_squared_error',
    'loss': 'dice',
    'nbepoch': 1000,
    'c_out': c_out,
    'nb_filters': 16,    
    'max_patience': 20    
        }

model = models.seg_model(params_train)
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
    print ('epoch: %s,  Current Learning Rate: %s' %(epoch,params_train['learning_rate']))
    seed = np.random.randint(0, 999999)

    # data augmentation
    bs2=128
    for k in range(0,X_train.shape[0],bs2):
        X_batch=X_train[k:k+bs]
        Y_batch=Y_train[k:k+bs]
        
        # preprocess 
        X_batch,Y_batch=utils.preprocess_XY(X_batch,Y_batch,param_prep)
        
        # augmentation
        X_batch,Y_batch=iterate_minibatches(X_batch,Y_batch,X_batch.shape[0],shuffle=False,augment=True)        
        model.fit(X_batch, Y_batch[:,0,:][:,np.newaxis,:], nb_epoch=1, batch_size=bs,verbose=0,shuffle=True)
    
    # evaluate on test and train data
    score_test=model.evaluate(*utils.preprocess_XY(X_test,Y_test,param_prep),verbose=0)
    score_train=model.evaluate(*utils.preprocess_XY(X_train,Y_train,param_prep),verbose=0)
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
    X,Y=utils.preprocess_XY(X_train[n1],Y_train[n1],param_prep)
else:
    X,Y=utils.preprocess_XY(X_test,Y_test,param_prep)

# prediction
Y_pred=model.predict(X)>.5    

plt.figure(figsize=(20,20))
n1=utils.disp_img_2masks(np.expand_dims(X[:,3,:],axis=1),Y_pred,Y,4,4,0)
plt.show()
#%%
# deply on test data
path2luna='./output/data/luna/test_luna_nodules_cin7.hdf5'
ff_test=h5py.File(path2luna,'w-')

print ss_test
t1=h5py.File(ss_test[0],'r')
for k in t1.keys():
    print k
    XY=t1[k]
    X0=XY[0]
    Y0=XY[1]
    X1=[]
    for k2 in range(0,X0.shape[0]-2,1):
        X1.append(X0[k2:k2+3])
    X1=np.stack(X1)
    Y_pred=model.predict(utils.preprocess(X1,param_prep))>.5
    ff_test[k]=Y_pred    
ff_test.close()        


Y_pred=utils.array_resize(Y_pred,(256,256))
X0p,Y0p=utils.preprocess_XY(X0[:,np.newaxis],Y0[:,np.newaxis],param_prep)
utils.disp_img_2masks(X0p,Y_pred,Y0p,4,5,0,range(0,20))

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