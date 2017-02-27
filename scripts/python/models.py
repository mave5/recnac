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


# model
def model(params):

    h=params['img_rows']
    w=params['img_cols']
    z=params['img_depth']
    lr=params['learning_rate']
    weights_path=params['weights_path']
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
    
    #load previous weights
    if weights_path:
        model.load_weights(weights_path)

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
    weights_path=params['weights_path']
    loss=params['loss']
    C=params['nb_filters']
    #num_labels=params['num_labels']
    c_out=params['c_out']
    
    
    
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
    conv10 = Convolution2D(c_out, 1, 1, activation='sigmoid')(conv10)

    model = Model(input=inputs, output=conv10)

    #load previous weights
    if weights_path:
        model.load_weights(weights_path)

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
from keras.layers import Activation, Reshape, Permute
#from keras.utils import np_utils


def classify_rnn(params):
    
    timestep=params['timestep']    
    h=params['h']
    w=params['w']
    c=params['c']
    nb_output=params['nb_output']
    loss=params['loss']
    optimizer=params['optimizer']
    
    # define model
    model = Sequential()
    
    model.add(TimeDistributed(Convolution2D(16, 3, 3, subsample=(2,2),border_mode='same',activation='relu'), input_shape=(timestep,c,h,w)))
    #model.add(TimeDistributed(Convolution2D(16, 3, 3, border_mode='same',activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),border_mode='valid')))
    
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same',activation='relu')))
    #model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same',activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),border_mode='valid')))
    
    model.add(TimeDistributed(Convolution2D(64, 3, 3, border_mode='same',activation='relu')))
    #model.add(TimeDistributed(Convolution2D(64, 3, 3, border_mode='same',activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),border_mode='valid')))
    
    model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same',activation='relu')))
    #model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same',activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),border_mode='valid')))
    
    #model.add(TimeDistributed(Convolution2D(256, 3, 3, border_mode='same',activation='relu')))
    #model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),border_mode='valid')))
    #model.add(TimeDistributed(Convolution2D(256, 3, 3, border_mode='same',activation='relu')))
    
    model.add(TimeDistributed(Flatten()))
    model.add(Activation('relu'))
    model.add(GRU(output_dim=100,return_sequences=True))
    model.add(GRU(output_dim=50,return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(nb_output, activation='sigmoid'))
    
    optimizer = RMSprop()
    model.compile(loss=loss, optimizer=optimizer)

    return model
    




    