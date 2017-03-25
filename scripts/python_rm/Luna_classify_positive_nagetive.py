#%% classify positive nodes and negative nodes

import numpy as np
import cv2
import time
import os
import matplotlib.pylab as plt
#from skimage import measure
import models
import utils
from keras import backend as K
from keras.utils import np_utils
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
import h5py
from glob import glob
from image import ImageDataGenerator
from keras.utils import np_utils
#%%
# settings

# path to dataset
path2luna_external="/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/"
path2subsets=path2luna_external+"subsets/"
#path2chunks=path2luna_external+"chunks/"
path2luna_internal="/media/mra/win71/data/misc/kaggle/datascience2017/LUNA2016/"
path2chunks=path2luna_internal+"chunks/"


subset_list=glob(path2chunks+'subset*.hdf5')
subset_list.sort()
print 'total subsets: %s' %len(subset_list)

#%%

# pre-processed data dimesnsion
z,h,w=64,64,64

# batch size
bs=8

# number of classes
num_classes=2

# fold number
foldnm=1

# seed point
seed = 2017
seed = np.random.randint(seed)

# exeriment name to record weights and scores
experiment='fold'+str(foldnm)+'_luna_classify_positive_negative'+'_hw_'+str(h)+'by'+str(w)+'_cin'+str(z)
print ('experiment:', experiment)

# checkpoint
weightfolder='./output/weights/'+experiment
if  not os.path.exists(weightfolder):
    os.makedirs(weightfolder)
    print ('weights folder created')

# data augmentation 
augmentation=True

# pre train
pre_train=True
#%%

########## log
import datetime
path2logs='./output/logs/'
now = datetime.datetime.now()
info='log_LunaClassifyPositiveNegative_'
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
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
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
    y=[]
    for ss in subsets:
        print ss
        ff=h5py.File(ss,'r')
        for k in ff.keys():
            print k
            X0=ff[k]['X'].value
            y0=ff[k]['y'].value
            X.append(X0)
            y.append(y0)
        ff.close()    
    X=np.vstack(X)    
    y=np.hstack(y)
    return X,y

#%%

print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)


# path to nfold train and test data
path2luna_train_test=path2chunks+'fold'+str(foldnm)+'_train_test_chunks.hdf5'
if os.path.exists(path2luna_train_test):
    ff_r=h5py.File(path2luna_train_test,'r')
    X_train=ff_r['X_train']
    y_train=ff_r['y_train'].value
    X_test=ff_r['X_test']
    y_test=ff_r['y_test'].value
    print 'hdf5 loaded'
else:    
    # train test split
    ss_test=subset_list.pop(foldnm)
    ss_train=subset_list
    print 'test:', ss_test
    print 'train:', ss_train
    
    # load images and masks 
    ff_w=h5py.File(path2luna_train_test,'w-')
    X_train,y_train=load_data(ss_train)   
    ff_w['X_train']=X_train
    ff_w['y_train']=y_train
    X_test,y_test=load_data([ss_test])   
    ff_w['X_test']=X_test
    ff_w['y_test']=y_test
    ff_w.close()
    print 'hdf5 saved!'
    
#%%
print('-'*30)
print('Creating and compiling model...')
print('-'*30)

# training params
params_train={
    'h': h,
    'w': w,
    'z': z,
    'c':1,           
    'learning_rate': 3e-5,
    'optimizer': 'Adam',
    #'loss': 'mean_squared_error',
    'loss': 'categorical_crossentropy',
    'nbepoch': 2000,
    'num_labels': num_classes,
    'nb_filters': 16,    
    'max_patience': 50    
        }

model = models.model_3d(params_train)
model.summary()

# path to weights
path2weights=weightfolder+"/weights.hdf5"

#%%

print ('training in progress ...')

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

# find nonzero diameters
pos_inds=np.nonzero(y_train)[0]
neg_inds=np.where(y_train==0)[0]

# convert to catogorical
#y_train=np.round(y_train/5).astype('uint8')
#y_test=np.round(y_test/5).astype('uint8')
y_train=y_train>0
y_test=y_test>0

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


for epoch in range(params_train['nbepoch']):

    print ('epoch: %s,  Current Learning Rate: %.1e' %(epoch,model.optimizer.lr.get_value()))
    seed = np.random.randint(0, 999999)

    # combine positives with random choice of negatives
    neg_rnd_inds=np.random.choice(neg_inds,len(pos_inds))    
    pos_neg_inds=np.append(pos_inds,neg_rnd_inds)
    np.random.shuffle(pos_neg_inds)
    
    # data augmentation
    bs2=64
    for k in range(0,len(pos_neg_inds),bs2):
        #print k
        batch_inds=list(pos_neg_inds[k:k+bs2])
        batch_inds.sort()
        try:
            X_batch=X_train[batch_inds]#[:,np.newaxis]
            y_batch=y_train[batch_inds]
        except:
            print 'skept this batch!'
            continue
        
        # augmentation
        X_batch,_=iterate_minibatches(X_batch,X_batch,X_batch.shape[0],shuffle=False,augment=True)        
        hist=model.fit(X_batch[:,np.newaxis], y_batch, nb_epoch=1, batch_size=bs,verbose=0,shuffle=True)
        #print 'partial loss:', hist.history['loss']
    
    # evaluate on test and train data
    #score_test=[]
    #for k2 in range(0,X_test.shape[0],bs):
    score_test=model.evaluate(np.array(X_test)[:,np.newaxis],y_test,verbose=0,batch_size=8)
    #score_test.append(tmp)
    #score_test=np.mean(np.array(score_test),axis=0)


    #if params_train['loss']=='dice': 
        #score_test=score_test[1]   
    score_train=hist.history['loss']
   
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