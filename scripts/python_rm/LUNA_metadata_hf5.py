#from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import csv
import h5py
import os
from glob import glob
import pandas as pd
import ntpath
from skimage.draw import circle
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x
#%%
H,W=512,512
    
#%%
    
# Helper function to get rows in data frame associated 
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)

def worldToVoxelCoord(worldCoord, origin, spacing):
     
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    
    return voxelCoord

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing    
    
#%%
############
#
# Getting list of image files
luna_path_save = "/media/mra/win71/data/misc/kaggle/datascience2017/LUNA2016/" 
#luna_path ="/media/mra/My Passport/Kaggle/datascience2017/LUNA2016/"
luna_path ="/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/"


luna_csv_path = luna_path+"CSVFILES/"


#%% loop over all subsets
for ss in range(0,10):
    subset=str(ss)
    luna_subset_path = luna_path+"subset"+subset+"/"
    file_list=glob(luna_subset_path+"*.mhd")
    print ('processing %s' %luna_subset_path)
    #
    # The locations of the nodes
    df_node = pd.read_csv(luna_csv_path+"annotations_excluded.csv")
    df_node.head()
    len(df_node["seriesuid"])
    mylist = list(set(df_node["seriesuid"]))
    
    print (len(df_node))
    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
    df_node = df_node.dropna()
    print (len(df_node))
    
    #####
    #
    # Looping over the image files
    #
    ff=h5py.File(luna_path_save+"subset"+subset+'_spacing.hdf5','w')
    for fcount, img_file in enumerate(file_list):
        print ntpath.basename(img_file)
        mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
        if mini_df.shape[0]>0: # some files may not have a nodule--skipping those 
            # load the data once
            _, origin, spacing = load_itk_image(img_file)
            print 'origin, spacing:', origin,spacing
                
            # store in hdf5 file                         
            seriesid=ntpath.basename(img_file)
            grp = ff.create_group(seriesid)
            grp['spacing']=spacing
            grp['origin']=origin
    ff.close()
#%%
