# usage: python classify_nodes.py nodes.npy 

import numpy as np
import pickle

from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF
import xgboost as xgb
from glob import glob
from skimage import measure
from sklearn import cross_validation
import os
import datetime
import pandas as pd
import ntpath
import matplotlib.pylab as plt
from skimage.draw import circle
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#%%

h,w=256,256
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


def getRegionMetricRow(fname = "nodules.npz"):
    # fname, numpy array of dimension [#slices, 1, 512, 512] containing the images
    #seg = np.load(fname)
    f1 = np.load(fname)
    seg=f1['Y'] # nodule 
    seg=coord2mask(seg,(h,w,R))
    #y=f1['y']
    nslices = seg.shape[0]

    #metrics
    totalArea = 0.
    avgArea = 0.
    maxArea = 0.
    avgEcc = 0.
    avgEquivlentDiameter = 0.
    stdEquivlentDiameter = 0.
    weightedX = 0.
    weightedY = 0.
    numNodes = 0.
    numNodesperSlice = 0.
    # crude hueristic to filter some bad segmentaitons
    # do not allow any nodes to be larger than 10% of the pixels to eliminate background regions
    maxAllowedArea = 0.10 * h * w
    
    areas = []
    eqDiameters = []
    for slicen in range(nslices):
        regions = getRegionFromMap(seg[slicen,0,:,:])
        for region in regions:
            if region.area > maxAllowedArea:
                continue
            totalArea += region.area
            areas.append(region.area)
            avgEcc += region.eccentricity
            avgEquivlentDiameter += region.equivalent_diameter
            eqDiameters.append(region.equivalent_diameter)
            weightedX += region.centroid[0]*region.area
            weightedY += region.centroid[1]*region.area
            numNodes += 1
    weightedX = weightedX / totalArea 
    weightedY = weightedY / totalArea
    avgArea = totalArea / numNodes
    avgEcc = avgEcc / numNodes
    avgEquivlentDiameter = avgEquivlentDiameter / numNodes
    stdEquivlentDiameter = np.std(eqDiameters)
    
    maxArea = max(areas)
    
    
    numNodesperSlice = numNodes*1. / nslices
    
    
    
    return np.array([avgArea,maxArea,avgEcc,avgEquivlentDiameter,\
                     stdEquivlentDiameter, weightedX, weightedY, numNodes, numNodesperSlice])


def createFeatureDataset(nodfiles):
    print ('wait ...')

    # ground truth data
    df_train = pd.read_csv('./output/submission/stage1_labels.csv')

    numfeatures = 9
    feature_array = np.zeros((len(nodfiles),numfeatures))
    truth_metric = np.zeros((len(nodfiles)))
    
    for i,nodfile in enumerate(nodfiles):
        print ('processing: %s, %s' %(i,nodfile))
        
        # extract subject id
        subject_id=ntpath.basename(nodfile)
        subject_id=subject_id.replace("_nodule.npz","")
        
        truth_metric[i] = int(df_train.cancer[df_train.id == subject_id])
        feature_array[i]= getRegionMetricRow(nodfile)
    
    return feature_array,truth_metric
    #np.save(nodfiles+"dataY.npy", truth_metric)
    #np.save(nodfiles+"dataX.npy", feature_array)

import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


#%%

# path to nodule files obtained from segmentation
path2noduls='./output/numpy/dsb/'
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

#%%
trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(X, Y, random_state=420, stratify=Y,
                                                                   test_size=0.1)
clf = xgb.XGBRegressor(max_depth=50,
                           n_estimators=1500,
                           min_child_weight=3,
                           learning_rate=0.02,
                           nthread=8,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=4242)
                           
clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss',early_stopping_rounds=5)
y_pred = clf.predict(val_x)
print("logloss",logloss(val_y, y_pred))
#%%
afasdasdasdasdasd
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

