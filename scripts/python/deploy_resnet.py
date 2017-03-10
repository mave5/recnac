# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 19:50:56 2017

@author: mra
"""

from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
import numpy as np

#%%
model = ResNet50(weights='imagenet',include_top=False)

img_path = 'elephant.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

#preds = model.predict(x)
#print('Predicted:', decode_predictions(preds))
# print: [[u'n02504458', u'African_elephant']]


features = model.predict(x)