import numpy as np
# functions
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D,Dropout
from keras.optimizers import Adam
#from keras.optimizers import SGD
#from keras.callbacks import ModelCheckpoint ,LearningRateScheduler
#from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Activation,Reshape,Permute,Flatten,Dense
#from keras.layers.advanced_activations import ELU
#from keras.models import Model
from keras import backend as K
#from keras.optimizers import Adam#, SGD
from keras.models import Sequential
#from funcs.image import ImageDataGenerator
#import h5py
import os
#%%
# model
def model(params):

    h=params['h']
    w=params['w']
    z=params['z']
    lr=params['learning_rate']
    #weights_path=params['weights_path']
    loss=params['loss']
    C=params['nb_filters']
    nb_output=params['nb_output']
    
    model = Sequential()
    model.add(Convolution2D(C, 3, 3, activation='relu',subsample=(1,1),border_mode='same', input_shape=(z, h, w)))

    N=5
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

    model.add(Dense(nb_output, activation='sigmoid'))
    #model.add(Dense(nb_output, activation='softmax'))
    
    #load previous weights
    #if weights_path:
        #model.load_weights(weights_path)

    model.compile(loss=loss, optimizer=Adam(lr))

    return model


#%%

from keras.models import Model

# smoothing factor when applying dice
smooth = 1.

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
def seg_model(params):

    h=params['h']
    w=params['w']
    c_in=params['c_in']
    lr=params['learning_rate']
    #weights_path=params['weights_path']
    loss=params['loss']
    C=params['nb_filters']
    #num_labels=params['num_labels']
    nb_output=params['nb_output']
    
    
    
    inputs = Input((c_in,h, w))
    conv1 = Convolution2D(C, 3, 3, activation='relu', subsample=(1,1),border_mode='same')(inputs)
    conv1 = Convolution2D(C, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(2*C, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(2*C, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(4*C, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(4*C, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(8*C, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(8*C, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(16*C, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(16*C, 3, 3, activation='relu', border_mode='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    # last layer of encoding    
    conv6 = Convolution2D(16*C, 3, 3, activation='relu', border_mode='same')(pool5)
    conv6 = Convolution2D(16*C, 3, 3, activation='relu', border_mode='same')(conv6)
    conv6 =Dropout(0.5)(conv6)
    
    # merge layers
    up6 = merge([UpSampling2D(size=(2, 2))(conv6), conv5], mode='concat', concat_axis=1)
    conv6 = Convolution2D(16*C, 3, 3, activation='relu', border_mode='same')(up6)
    #conv6 = Convolution2D(8*C, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv4], mode='concat', concat_axis=1)
    conv7 = Convolution2D(8*C, 3, 3, activation='relu', border_mode='same')(up7)
    #conv7 = Convolution2D(4*C, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv3], mode='concat', concat_axis=1)
    conv8 = Convolution2D(4*C, 3, 3, activation='relu', border_mode='same')(up8)
    #conv8 = Convolution2D(2*C, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv2], mode='concat', concat_axis=1)
    conv9 = Convolution2D(2*C, 3, 3, activation='relu', border_mode='same')(up9)
    #conv9 = Convolution2D(C, 3, 3, activation='relu', border_mode='same')(conv9)

    up10 = merge([UpSampling2D(size=(2, 2))(conv9), conv1], mode='concat', concat_axis=1)
    conv10 = Convolution2D(C, 3, 3, activation='relu', border_mode='same')(up10)
    #conv9 = Convolution2D(C, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(C, 3, 3, activation='relu', border_mode='same')(conv10)
    conv10 = Convolution2D(nb_output, 1, 1, activation='sigmoid')(conv10)

    model = Model(input=inputs, output=conv10)

    #load previous weights
    #if weights_path:
        #model.load_weights(weights_path)

    if loss=='dice':
        model.compile(optimizer=Adam(lr), loss=dice_coef_loss, metrics=[dice_coef])
    else:
        #model.compile(loss='binary_crossentropy', optimizer=Adam(lr))
        model.compile(loss=loss, optimizer=Adam(lr))
    
    return model

#%%

# model
def seg_encode_decode(params):

    h=params['h']
    w=params['w']
    c_in=params['c_in']
    lr=params['learning_rate']
    weights_path=params['weights_path']
    lossfunc=params['loss']
    C=params['nb_filters']
    c_out=params['c_out']
    
    model = Sequential()
    
    model.add(Convolution2D(C, 3, 3, activation='relu',border_mode='same', input_shape=(c_in, h, w)))

    N=6
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
    model.add(Convolution2D(c_out, 1, 1, activation='sigmoid', border_mode='same'))                            
    

    #load previous weights
    if weights_path:
        model.load_weights(weights_path)

    if lossfunc=='dice':
        model.compile(optimizer=Adam(lr), loss=dice_coef_loss, metrics=[dice_coef])
    else:
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr))

    return model
    
#%%

from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.optimizers import RMSprop, Adadelta
from keras.layers import Input, Dropout
#from keras.layers import Activation, Reshape, Permute
#from keras.utils import np_utils


def classify_rnn(params):
    
    timesteps=params['timesteps']    
    h=params['h']
    w=params['w']
    z=params['z']
    lr=params['learning_rate']
    nb_output=params['nb_output']
    loss=params['loss']
    C=params['nb_filters']
    #optimizer=params['optimizer']

    # define model
    model = Sequential()
    
    model.add(TimeDistributed(Convolution2D(C, 3, 3, subsample=(1,1),border_mode='same',activation='relu'), input_shape=(timesteps,z,h,w)))
    #model.add(TimeDistributed(Convolution2D(16, 3, 3, border_mode='same',activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),border_mode='valid')))
    
    model.add(TimeDistributed(Convolution2D(2*C, 3, 3, border_mode='same',activation='relu')))
    #model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same',activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),border_mode='valid')))
    
    model.add(TimeDistributed(Convolution2D(4*C, 3, 3, border_mode='same',activation='relu')))
    #model.add(TimeDistributed(Convolution2D(64, 3, 3, border_mode='same',activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),border_mode='valid')))
    
    model.add(TimeDistributed(Convolution2D(8*C, 3, 3, border_mode='same',activation='relu')))
    #model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same',activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),border_mode='valid')))
    
    #model.add(TimeDistributed(Convolution2D(256, 3, 3, border_mode='same',activation='relu')))
    #model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),border_mode='valid')))
    #model.add(TimeDistributed(Convolution2D(256, 3, 3, border_mode='same',activation='relu')))
    
    model.add(TimeDistributed(Flatten()))
    model.add(Dropout(0.1))
    #model.add(Activation('relu'))
    model.add(GRU(output_dim=100,return_sequences=False))
    #model.add(GRU(output_dim=100,return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(nb_output, activation='sigmoid'))
    
    optimizer = RMSprop(lr)
    #optimizer = Adam(lr)
    model.compile(loss=loss, optimizer=optimizer)

    return model
    
#%%


# model
def vgg_model(params):

    h=params['h']
    w=params['w']
    z=params['z']
    lr=params['learning_rate']
    #weights_path=params['weights_path']
    loss=params['loss']
    C=params['nb_filters']
    nb_output=params['nb_output']
    
    model = Sequential()
    model.add(Convolution2D(C, 3, 3, activation='relu',subsample=(1,1),border_mode='same', input_shape=(z, h, w)))

    N=5
    for k in range(1,N):
        C1=np.min([2**k*C,512])
        model.add(Convolution2D(C1, 3, 3, activation='relu', subsample=(1,1), border_mode='same'))              
        model.add(Convolution2D(C1, 3, 3, subsample=(1,1), activation='relu', border_mode='same'))              
        model.add(Convolution2D(C1, 3, 3, subsample=(1,1), activation='relu', border_mode='same'))                      
        model.add(MaxPooling2D(pool_size=(2, 2)))

        
    #model.add(Convolution2D(C1, 3, 3, activation='relu', border_mode='same'))              
    #model.add(Convolution2D(C1, 3, 3, activation='relu', border_mode='same'))              
    #model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dropout(0.5))
    #model.add(Dense(100, activation='relu'))
    #model.add(Dropout(0.1))

    model.add(Dense(nb_output, activation='sigmoid'))
    
    #load previous weights
    #if weights_path:
        #model.load_weights(weights_path)

    model.compile(loss=loss, optimizer=Adam(lr))

    return model
#%%

#from keras.layers import merge, Input
#from keras.layers import Dense, Activation, Flatten
#from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
#from keras.models import Model
#from keras.preprocessing import image
#import keras.backend as K
#from keras.utils.layer_utils import convert_all_kernels_in_model
#from keras.utils.data_utils import get_file
#from imagenet_utils import decode_predictions, preprocess_input

def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def resnet_model(params):    
    
    h=params['h']
    w=params['w']
    z=params['z']
    lr=params['learning_rate']
    loss=params['loss']
    nb_output=params['nb_output']
    bn_axis = 1
    
    img_input = Input(shape=(z,h,w))
    
    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(1, 1), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((8, 8), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(nb_output, activation='sigmoid', name='fc1000')(x)

    model = Model(img_input, x)    
    
    model.compile(loss=loss, optimizer=Adam(lr))
    
    return model
    
#%%


def classify_rnn2(params):
    
    timesteps=params['timesteps']    
    h=params['h']
    w=params['w']
    z=params['z']
    lr=params['learning_rate']
    nb_output=params['nb_output']
    loss=params['loss']
    C=params['nb_filters']
    #optimizer=params['optimizer']

  
    inputs = Input((z,h, w))
    conv1 = Convolution2D(C, 3, 3, activation='relu', subsample=(1,1),border_mode='same')(inputs)
    conv1 = Convolution2D(C, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(2*C, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(2*C, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(4*C, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(4*C, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(8*C, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(8*C, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(16*C, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(16*C, 3, 3, activation='relu', border_mode='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    # last layer of encoding    
    conv6 = Convolution2D(16*C, 3, 3, activation='relu', border_mode='same')(pool5)
    conv6 = Convolution2D(16*C, 3, 3, activation='relu', border_mode='same')(conv6)
    conv6 =Dropout(0.5)(conv6)    
    
    model_encoder = Model(input=inputs, output=conv6)
    
    model = Sequential()
    model.add(TimeDistributed(Convolution2D(C, 3, 3, subsample=(1,1),border_mode='same',activation='relu'), input_shape=(timesteps,z,h,w)))
    model.add(TimeDistributed(model_encoder))
    
   
    model.add(TimeDistributed(Flatten()))
    model.add(Dropout(0.1))
    #model.add(Activation('relu'))
    model.add(GRU(output_dim=100,return_sequences=False))
    #model.add(GRU(output_dim=100,return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(nb_output, activation='sigmoid'))
    
    optimizer = RMSprop(lr)
    #optimizer = Adam(lr)
    model.compile(loss=loss, optimizer=optimizer)

    return model
#%%

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
# model
def model_pretrain(params):

    h=params['h']
    w=params['w']
    c_in=params['c_in']
    lr=params['learning_rate']
    path2segweights=params['path2segweights']
    loss=params['loss']
    C=params['nb_filters']
    nb_output=params['nb_output']
    c_out=params['nb_output']
    
    # seg model    
    inputs = Input((c_in,h, w))
    conv1 = Convolution2D(C, 3, 3, activation='relu', subsample=(1,1),border_mode='same')(inputs)
    conv1 = Convolution2D(C, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(2*C, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(2*C, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(4*C, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(4*C, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(8*C, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(8*C, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(16*C, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(16*C, 3, 3, activation='relu', border_mode='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    # last layer of encoding    
    conv6 = Convolution2D(16*C, 3, 3, activation='relu', border_mode='same')(pool5)
    conv6 = Convolution2D(16*C, 3, 3, activation='relu', border_mode='same')(conv6)
    conv6 =Dropout(0.5)(conv6)
    conv6=Flatten()(conv6)    

    half_seg_model = Model(input=inputs, output=conv6)
    #make_trainable(half_seg_model,False)
    #half_seg_model.summary()
    
    # get the complete segmentation model
    seg_model2=seg_model(params)
    #seg_model2.summary()
    
    # load weights into segmentation model
    if os.path.exists(path2segweights):
        seg_model2.load_weights(path2segweights)
        print 'pretrain weights loaded into seg model!'
    else:
        raise IOError('weights not found!')
    
    # load weights and vrify
    for k in range(18):
        # get weights
        wk=seg_model2.layers[k].get_weights()
        # set weights
        half_seg_model.layers[k].set_weights(wk)
        wk2=half_seg_model.layers[k].get_weights()
        if len(wk)>0:
            print 'weight sum in layer %s: %s, %s' %(k,np.sum(wk[0]),np.sum(wk2[0]))
    
    # define rnn model
    model=Sequential()   
    model.add(half_seg_model)    
    model.add(Dense(100,activation='relu'))
    model.add(Dense(nb_output,activation='sigmoid'))    

    if loss=='dice':
        model.compile(optimizer=Adam(lr), loss=dice_coef_loss, metrics=[dice_coef])
    else:
        #model.compile(loss='binary_crossentropy', optimizer=Adam(lr))
        model.compile(loss=loss, optimizer=Adam(lr))
    
    return model
#%%


def rnn_pretrain(params):
    
    timesteps=params['timesteps']    
    h=params['h']
    w=params['w']
    z=params['z']
    lr=params['learning_rate']
    nb_output=params['nb_output']
    loss=params['loss']
    C=params['nb_filters']
    #optimizer=params['optimizer']
    path2segweights=params['path2segweights']
    c_in=params['c_in']
    
    # define half of segmentation model    
    inputs = Input((c_in,h, w))
    conv1 = Convolution2D(C, 3, 3, activation='relu', subsample=(1,1),border_mode='same')(inputs)
    conv1 = Convolution2D(C, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(2*C, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(2*C, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(4*C, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(4*C, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(8*C, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(8*C, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(16*C, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(16*C, 3, 3, activation='relu', border_mode='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    # last layer of encoding    
    conv6 = Convolution2D(16*C, 3, 3, activation='relu', border_mode='same')(pool5)
    conv6 = Convolution2D(16*C, 3, 3, activation='relu', border_mode='same')(conv6)
    conv6 =Dropout(0.5)(conv6)
    
    half_seg_model = Model(input=inputs, output=conv6)
    # freeze the weights    
    make_trainable(half_seg_model,False)


    # get the complete segmentation model
    seg_model2=seg_model(params)
    seg_model2.summary()
    
    # load weights into segmentation model
    if os.path.exists(path2segweights):
        seg_model2.load_weights(path2segweights)
        print 'pretrain weights loaded into seg model!'
    else:
        raise IOError('weights not found!')
    
    # load weights and vrify
    for k in range(18):
        # get weights
        wk=seg_model2.layers[k].get_weights()
        # set weights
        half_seg_model.layers[k].set_weights(wk)
        wk2=half_seg_model.layers[k].get_weights()
        if len(wk)>0:
            print 'weight sum in layer %s: %s, %s' %(k,np.sum(wk[0]),np.sum(wk2[0]))
    
    # define rnn model
    model=Sequential()   
    model.add(TimeDistributed(half_seg_model,input_shape=(timesteps,z,h,w)))
    
    # check for freezed weights    
    wk=model.layers[0].get_weights()
    print np.sum([np.sum(x) for x in zip(wk)])
    
    model.add(TimeDistributed(Flatten()))
    #model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(GRU(output_dim=100,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(output_dim=50,return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(nb_output, activation='sigmoid'))

    rmsprop = RMSprop(lr)
    model.compile(loss=loss, optimizer=rmsprop)
    
    return model
#%%
    
from keras.layers import Convolution3D, MaxPooling3D

# model
def model_3d(params):
    h=params['h']
    w=params['w']
    z=params['z']
    c=params['c']
    
    lr=params['learning_rate']
    loss=params['loss']
    C=params['nb_filters']
    num_labels=params['num_labels']
    
    
    model = Sequential()    
    model.add(Convolution3D(C, 3,3, 3, activation='relu',subsample=(1,1,1),border_mode='same', input_shape=(c,z, h, w)))
    model.add(MaxPooling3D(pool_size=(2,2, 2)))
    
    N=5
    for k in range(1,N):
        ck=min([2**k*C,512])
        model.add(Convolution3D(ck, 3,3, 3, activation='relu', border_mode='same'))              
        model.add(Convolution3D(ck, 3,3, 3, activation='relu', border_mode='same'))              
        model.add(MaxPooling3D(pool_size=(2,2, 2)))
    
    model.add(Convolution3D(ck, 3,3, 3, activation='relu', border_mode='same'))              
    
    #model.add(Convolution2D(2**k*C, 3, 3, activation='relu', border_mode='same'))              
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_labels, activation='softmax'))
    #model.add(Dense(num_labels, activation='sigmoid'))
    
    model.compile(loss=loss, optimizer=Adam(lr))

    return model

    
    