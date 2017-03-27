import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import cv2
import utils
import scipy.ndimage as ndimage
from sklearn.cross_validation import KFold
from skimage import measure, morphology, segmentation
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import h5py
#%matplotlib inline
p = sns.color_palette()
import time

# get package versions
def get_version(*vars):
    for var in vars:
        module = __import__(var)    
        print '%s: %s' %(var,module.__version__)
    
# package version    
get_version('numpy','matplotlib','cv2','sklearn','skimage','scipy')
#%%

path2data='../sample_images/'
path2data='/media/mra/My Passport/Kaggle/datascience2017/data/stage1/'
patients=os.listdir(path2data)
patients.sort()
print len(patients)

# resize
h,w=512,512
#%%

# Load dicom files
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness

    RefDs=slices[0]
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))        
    return slices,ConstPixelSpacing
    
    
#%%

print 'wait ...'
nb_dicoms=[]
for d in patients:
    #print("Patient '{}' has {} scans".format(d, len(os.listdir(path2data + d))))
    nb_dicoms.append(len(os.listdir(path2data + d)))

utils.array_stats(nb_dicoms)    
print('Total patients {} Total DCM files {}'.format(len(patients), len(glob.glob(path2data+'*/*.dcm'))))

plt.figure(figsize=((10,5)))
plt.hist(nb_dicoms, color=p[2])
plt.ylabel('Number of patients')
plt.xlabel('DICOM files')
plt.title('Histogram of DICOM count per patient')
plt.show()


#%%

df_train = pd.read_csv('../stage1_labels.csv')
print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
df_train.head()

path2spacing='/media/mra/win71/data/misc/kaggle/datascience2017/data/dsb_spacing.hdf5'
ff_dsb=h5py.File(path2spacing,'w-')

for k,p_id in enumerate(df_train.id):
    _,spcaing=load_scan(path2data+p_id)
    print k,p_id,spcaing
    ff_dsb[p_id]=spcaing
ff_dsb.close()
print 'spacing saved!'

# verify
ff_dsb=h5py.File(path2spacing,'r')
print 'total:', len(ff_dsb)
ff_dsb.close()

#%%

# dsb test

df_train = pd.read_csv('../stage1_submission.csv')
print('Number of training patients: {}'.format(len(df_train)))
#print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
df_train.head()

path2spacing='/media/mra/win71/data/misc/kaggle/datascience2017/data/dsbtest_spacing.hdf5'

ff_dsb=h5py.File(path2spacing,'w-')

for k,p_id in enumerate(df_train.id):
    _,spcaing=load_scan(path2data+p_id)
    print k,p_id,spcaing
    ff_dsb[p_id]=spcaing
ff_dsb.close()
print 'spacing saved!'

# verify
ff_dsb=h5py.File(path2spacing,'r')
print 'total:', len(ff_dsb)
ff_dsb.close()
#%%

path2spacing='/media/mra/win71/data/misc/kaggle/datascience2017/data/dsb_spacing.hdf5'
path2spacing2='/media/mra/win71/data/misc/kaggle/datascience2017/data/dsb_spacing2.hdf5'
ff_dsb=h5py.File(path2spacing,'r')
ff_dsb2=h5py.File(path2spacing2,'w-')
print 'total:', len(ff_dsb)

# reverse to z y z
for key in ff_dsb.keys():
    spacing=ff_dsb[key].value
    print key,spacing
    ff_dsb2[key]=spacing[::-1]
ff_dsb2.close()
ff_dsb.close()
print 'spacing reverse saved!'

#%%

path2spacing='/media/mra/win71/data/misc/kaggle/datascience2017/data/dsbtest_spacing.hdf5'
path2spacing2='/media/mra/win71/data/misc/kaggle/datascience2017/data/dsbtest_spacing2.hdf5'

ff_dsb=h5py.File(path2spacing,'r')
ff_dsb2=h5py.File(path2spacing2,'w-')
print 'total:', len(ff_dsb)

# reverse to z y z
for key in ff_dsb.keys():
    spacing=ff_dsb[key].value
    print key,spacing
    ff_dsb2[key]=spacing[::-1]
ff_dsb2.close()
ff_dsb.close()
print 'spacing reverse saved!'




