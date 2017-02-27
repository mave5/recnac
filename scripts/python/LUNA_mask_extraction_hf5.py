from __future__ import print_function, division
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
#Some helper functions

def make_mask(center,diam,z,width,height,spacing,origin):
    '''
Center : centers of circles px -- list of coordinates x,y,z
diam : diameters of circles px -- diameter
widthXheight : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5]) 
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    return(mask)

def matrix2int16(matrix):
    ''' 
matrix must be a numpy array NXN
Returns uint16 version
    '''
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))
    
#####################
#
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
    

def coord2mask(coords,params):
    h,w,diam=params
    Y=np.zeros((3,h,w),dtype='uint8')
    img = np.zeros((h, w), dtype=np.uint8)
    c,r=coords[:2]
    radius=diam
    rr, cc = circle(r,c,radius)
    img[rr, cc] = 1
    Y[:]=img
    return Y

    
#%%
############
#
# Getting list of image files
luna_path_save = "/media/mra/win7/data/misc/kaggle/datascience2017/LUNA2016/" 
luna_path ="/media/mra/My Passport/Kaggle/datascience2017/LUNA2016/"


luna_csv_path = luna_path+"CSVFILES/"
#output_path = "./output/numpy/luna/subset/"
#if not os.path.isdir(output_path):
#    os.mkdir(output_path)
#%% loop over all subsets
for ss in range(0,10):
    subset=str(ss)
    luna_subset_path = luna_path+"subset"+subset+"/"
    file_list=glob(luna_subset_path+"*.mhd")
    print ('processing %s' %luna_subset_path)
    #
    # The locations of the nodes
    df_node = pd.read_csv(luna_csv_path+"annotations.csv")
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
    ff=h5py.File(luna_path_save+"subset"+subset+'.hdf5','w-')
    for fcount, img_file in enumerate(tqdm(file_list)):
        mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
        if mini_df.shape[0]>0: # some files may not have a nodule--skipping those 
            # load the data once
            itk_img = sitk.ReadImage(img_file) 
            img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
            masks = np.zeros_like(img_array,dtype='uint8')#np.ndarray([3,height,width],dtype=np.uint8)
            num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
            origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
            # go through all nodes (why just the biggest?)
            for node_idx, cur_row in mini_df.iterrows():       
                node_x = cur_row["coordX"]
                node_y = cur_row["coordY"]
                node_z = cur_row["coordZ"]
                diam = cur_row["diameter_mm"]
                #print (diam)
                # just keep 3 slices
                #imgs = np.ndarray([3,height,width],dtype=np.float32)
                #masks = np.ndarray([3,height,width],dtype=np.uint8)
                center = np.array([node_x, node_y, node_z])   # nodule center
                #v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
                v_center=np.round(worldToVoxelCoord(center, origin, spacing))
                if v_center[2]+2<num_z:                
                    i_z=np.arange(int(v_center[2])-1,int(v_center[2])+2)                
                else:
                   i_z=np.arange(v_center[2]-2,num_z)                
                Y=coord2mask(v_center,(H,W,diam/spacing[0]))   
                # nodule masks                 
                masks[i_z]=Y                                     
                
            # store in hdf5 file                         
            img_file_base=ntpath.basename(img_file)
            XY=[img_array,masks]
            ff[img_file_base]=XY 
    ff.close()
#%%
