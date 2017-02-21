import numpy as np
# functions
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D,Dropout
from keras.optimizers import Adam
#from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint ,LearningRateScheduler
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
