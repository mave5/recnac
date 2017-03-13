import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import cross_validation
import datetime
import utils
import cv2
import random
import time
import os
import matplotlib.pylab as plt
import scipy as sp
import h5py    
import models
#from keras.preprocessing.image import ImageDataGenerator
from image import ImageDataGenerator
#%%

root_data='/media/mra/win7/data/misc/kaggle/datascience2017/data/'
path2dsb=root_data+'dsb.hdf5'
path2csv=root_data+'stage1_labels.csv'
path2logs='./output/logs/'

# resize
H,W=512,512

# batch size
bs=1

c_in=7

# trained data dimesnsion
h,w=256,256

# time step
timestep=20

# exeriment name to record weights and scores
experiment='dsb_rnn_classify'+'_hw_'+str(h)+'by'+str(w)+'_cin'+str(c_in)
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



def extract_dsb(ids,augmentation=False):

    #X=[]    
    y=[]
    Y_seg=[]
    for id in ids:    
        # read h5 file
        f2=h5py.File(path2dsb,'r')
        X1=f2[id]
        n=X1.shape[0]
        X1=np.append(X1,np.zeros((c_in-n%c_in,H,W),dtype='int16'),axis=0)
        #print X1.shape
        n=X1.shape[0]
        X1=np.reshape(X1,(n/c_in,c_in,H,W))
        #print X1.shape
        #X.append(X1)
        y1=f2[id].attrs['cancer']
        y.append(y1)

        # preprocess        
        X1=utils.preprocess(X1,param_prep)

        # augmentation
        if augmentation:
            X1,_=iterate_minibatches(X1,X1,X1.shape[0],shuffle=False,augment=True)                
              
        # obtain nodules 
        th_area=10      
        Yp_seg=seg_model.predict(X1)>0.5
        nz_yp=np.where(np.sum(Yp_seg,axis=(1,2,3))>th_area)
        print 'cancer %s, number of non-zero masks:  %s' %(y1,len(nz_yp[0]))
        
        # pick non zeros
        X1=X1[nz_yp]        
        Yp_seg=Yp_seg[nz_yp]
        
        # concat image with mask
        #Yp_seg=np.append(X1,Yp_seg,axis=1)
        #for i in range(X1.shape[0]):
            #for j in range(X1.shape[1]):
                #X1[i,j,:]=np.multiply(X1[i,j,:],Yp_seg[i,0])

        #Yp_seg=X1
        Y_seg.append(Yp_seg)
    
    # prepare for RNN: 5D array    
    Yp_seg=pad4rnn(Y_seg,timestep)

    return Yp_seg,np.array(y, dtype=np.uint8)
    #return X,np.array(y, dtype=np.uint8)
    

def pad4rnn(X,timestep):
    c,h,w=X[0].shape[1:]
    Xt=np.zeros((len(X),timestep,c,h,w),'float32')
    for k in range(len(X)):
        # pre-prate for RNN        
        if X[k].shape[0]>timestep:
            Xt[k]=X[k][:timestep]
        else:
            Xt[k]=np.append(np.zeros((timestep-X[k].shape[0],c,h,w),'float32'),X[k],axis=0)
    
    return Xt

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
non_cancer=df_train[df_train.cancer==0].id
cancer=df_train[df_train.cancer==1].id
print 'total non cancer:%s, total cancer:%s' %(len(non_cancer),len(cancer))

y_nc=np.zeros(len(non_cancer),'uint8')
y_c=np.ones(len(cancer),'uint8')

# train val split
#path2trainvalindx=weightfolder+'/dsb_train_val_index'
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


#%%

# nodule segmentation network

# training params
params_seg={
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
    'c_out': 1,
    'nb_filters': 16,    
    'max_patience': 30    
        }

seg_model=models.seg_model(params_seg)

seg_model.summary()

# path to weights
path2segweights=weightfolder+"/weights_seg.hdf5"
if  os.path.exists(path2segweights):
    seg_model.load_weights(path2segweights)
    print 'weights loaded!'
else:
    raise IOError
    

#%%

# training params
params_train={
        'h': h,
        'w': w,
        'c': 1,
        'timestep': timestep,
        'optimizer': 'RMSprop()',
        'learning_rate': 1e-5,
        'optimizer': 'rmsprop',
        'loss': 'binary_crossentropy',
        #'loss': 'mean_squared_error',
        'nbepoch': 1000,
        'nb_output': 1,
        'nb_filters': 16,    
        'max_patience': 30    
        }
model_rnn=models.classify_rnn(params_train)
model_rnn.summary()

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
        model_rnn.load_weights(path2weights)
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
    print ('epoch: %s,  Current Learning Rate: %.1e' %(epoch,model_rnn.optimizer.lr.get_value()))
    seed = np.random.randint(0, 999999)
    trn_ids_pick=random.sample(trn_ids,256)

    bs2=8*bs
    for t1 in range(0,len(trn_ids_pick),bs2):
        trn_id_batch=trn_ids_pick[t1:t1+bs2]                
        print t1
    
        # extract nodule masks 
        X_batch,y_batch=extract_dsb(trn_id_batch,False)    

        # fit model to data
        #class_weight={0:0.25,1:1.}
        model_rnn.fit(X_batch, np.array(y_batch), nb_epoch=1, batch_size=bs2,verbose=0,shuffle=False)    

        
    # evaluate on test and train data
    X_test,y_test=extract_dsb(val_ids,False)    
    score_test=model_rnn.evaluate(X_test,np.array(y_test),verbose=0)
    
    score_train=model_rnn.evaluate(X_batch,np.array(y_batch),verbose=0)
    
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
            model_rnn.save_weights(path2weights)            
            
        # learning rate schedule
        if score_test<=previous_score:
            #print "Incrementing Patience."
            patience += 1
    else:
        if (score_test<=best_score):
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!! viva, improvement!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
            best_score = score_test
            patience = 0
            model_rnn.save_weights(path2weights)            
            
        # learning rate schedule
        if score_test>previous_score:
            #print "Incrementing Patience."
            patience += 1
        
    
    if patience == params_train['max_patience']:
        params_train['learning_rate'] = params_train['learning_rate']/2
        print ("Upating Current Learning Rate to: ", params_train['learning_rate'])
        model_rnn.optimizer.lr.set_value(params_train['learning_rate'])
        print ("Loading the best weights again. best_score: ",best_score)
        model_rnn.load_weights(path2weights)
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
    model_rnn.load_weights(path2weights)
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

#score_test=model_rnn.evaluate(utils.preprocess(X_test,param_prep),y_test,verbose=0)
score_test=model_rnn.evaluate(X_test,y_test,verbose=0)
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

for z3 in range(X_batch.shape[1]):
    plt.subplot(4,5,z3+1)
    plt.imshow(X_batch[0,z3],cmap='gray')
    

