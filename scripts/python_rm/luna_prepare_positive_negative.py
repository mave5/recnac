#from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
#import csv
import scipy.ndimage
import h5py
import os
import utils
from glob import glob
import pandas as pd
import ntpath
#from skimage.draw import circle

#%%
H,W=512,512

# crop size
wc=64

# read images and nodes from subset
path2luna_external="/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/"
path2subsets=path2luna_external+"subsets/"
path2chunks=path2luna_external+"chunks/"

path2luna_internal="/media/mra/win71/data/misc/kaggle/datascience2017/LUNA2016/"
path2subsets_rescale=path2luna_internal+'subset_rescale/'

# list of subsets
subset_list=glob(path2subsets_rescale+'subset*.hdf5')
subset_list.sort()
print 'total subsets: %s' %len(subset_list)

# read annotations csv
path2annotations=path2luna_internal+'CSVFILES/annotations_excluded.csv'

df_node = pd.read_csv(path2annotations)
print 'dataset size:', len(df_node["seriesuid"])
mylist = list(set(df_node["seriesuid"]))
#print 'sample series id:', mylist[0]
df_node.head()

#%%

def normalize(image):
    ####### normalization
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def worldToVoxelCoord(worldCoord, origin, spacing):
     
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    
    return voxelCoord

def resample_vCord(voxelCoord, spacing, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    
    resize_factor = spacing / new_spacing
    new_voxelCoord = voxelCoord * resize_factor
    new_voxelCoord = np.round(new_voxelCoord).astype('int16')
    #real_resize_factor = new_shape / voxelCoord
    #new_spacing = spacing / real_resize_factor
    return new_voxelCoord
    
# get positive and negative nodes
def get_nodes(Xr,mini_df,wc):
    # wc : crop size
    # Xr: stack of resampled images: N*H*W 
    # mini_df: coordinates of positive and negative nodes
    
    w=wc/2 # crop size
    re_voxelCoord=[] # voxel coordinate in resampled images
    diams=[] # to store diameters
    Xc=[] # to store crops
    
    # loop over coordinates
    for node_idx, cur_row in mini_df.iterrows():       
        node_x = cur_row["coordX"]
        node_y = cur_row["coordY"]
        node_z = cur_row["coordZ"]
        diam = cur_row["diameter_mm"]
        worldCoord=[node_z,node_y,node_x]        
        # check for nodules            
        if diam>0:
            # convert to voxel coordinate
            voxelCoord=np.round(worldToVoxelCoord(worldCoord, origin, spacing)).astype('uint16')
            # voxel coordinate in the resampled domain                
            re_vc=resample_vCord(voxelCoord,spacing) 
            re_voxelCoord.append(re_vc)
            diams.append(diam)
        else:
            # convert to voxel coordinate
            voxelCoord=np.round(worldToVoxelCoord(worldCoord, origin, spacing)).astype('uint16')
            # voxel coordinate in the resampled domain                
            re_vc=resample_vCord(voxelCoord,spacing) 
            re_voxelCoord.append(re_vc)
            diam=0
            diams.append(diam)
            
            
    # find unique nodes
    re_voxelCoord=np.array(re_voxelCoord)    
    _,ind_nodes=np.unique(re_voxelCoord[:,0],return_index=True)
    re_voxelCoord=re_voxelCoord[ind_nodes]
    diams=np.array(diams)[ind_nodes]

    # loop over all unique nodes
    for k1 in range(re_voxelCoord.shape[0]):                
        # crop coords
        zyx_1 = re_voxelCoord[k1] - w # Attention: Z, Y, X
        zyx_2 = re_voxelCoord[k1] + w 
        #print zyx_1.dtype, zyx_1,zyx_2                    
    
        for k in range(3):
            if zyx_1[k]<0:
                zyx_1[k]=0
                zyx_2[k]=2*w
            if zyx_2[k]>Xr.shape[k]:
                zyx_1[k]=Xr.shape[k]-2*w
                zyx_2[k]=Xr.shape[k]       
        #print zyx_1,zyx_2                    
        
        # crop 2w*2w*2w
        Xc1 = Xr[ zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2] ]
        Xc.append(Xc1)
        #print Xc1.shape
    Xc=np.stack(Xc)    

    return Xc,diams

#%%

# loop over all subsets
for ss in subset_list:
    if ('lung' in ss) or ('spacing' in ss):
        continue
    print 'subset:', ss
    print '-'*50
    
    # save chunks
    base_nm=ntpath.basename(ss).replace('rescale','chunks')
    ff_chunks=h5py.File(path2chunks+base_nm,'w-')
    
    # hdf5 for re-sampled images
    ff_Xr=h5py.File(ss,'r')
    
    # hdf5 for re-sampled lung masks
    ff_Yr=h5py.File(ss.replace('rescale','lung_rescale'),'r')
    
    # hdf5 for spacing file
    ff_spacing=h5py.File(ss.replace('rescale','spacing'),'r')
    
    # loop over all ids in each subset
    for key in ff_Xr.keys():
        print 'patient id: %s' %key
        Xr=np.array(ff_Xr[key],'float32')
        Xr=normalize(Xr)
        #utils.array_stats(Xr)
        Yr=ff_Yr[key]
        #utils.array_stats(Yr)
        
        # extract lung only
        Xr=Xr*Yr
        utils.array_stats(Xr)
        
        # read spacing and origin
        spacing=ff_spacing[key]['spacing'].value
        origin=ff_spacing[key]['origin'].value        

        #get all nodules associate with file
        mini_df = df_node[df_node["seriesuid"]==key[:-4]] 
        
        # loop over all nodes: positive and negative
        # get positive nodes 
        Xc,diams=get_nodes(Xr,mini_df,wc)
        print Xc.shape,diams.shape
        
        # save chunck
        grp=ff_chunks.create_group(key)
        grp['X']=Xc
        grp['y']=diams
        print 'chuncks saved!'        
        print '-'*50

# close hdf5
ff_chunks.close()
#plt.imshow(Xc[7,31],cmap='gray')