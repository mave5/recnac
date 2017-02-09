import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
import time
import matplotlib.pylab as plt
from sklearn.externals import joblib
#%%

#path2data='/media/mra/win7/data/misc/kaggle/datascience2017/notebooks/output/data/dsb/'
path2data='/media/mra/win7/data/misc/kaggle/datascience2017/notebooks/output/data/dsb_test/'
file_list=glob(path2data+'*.npz')
print 'tatal subjects: %s' %(len(file_list))
#%%

# path to csv file to save scores
path2scorescsv = path2data+'/log.csv'
first_row = 'subject number,file name'
with open(path2scorescsv, 'w+') as f:
    f.write(first_row + '\n')
    
subject_num=0
for subject_num, img_file in enumerate(file_list):
    start_time=time.time()
    # I ran into an error when using Kmean on np.float16, so I'm using np.float64 here
    #imgs_to_process = np.load(img_file).astype(np.float64) 

    # subject number   
    #subject_num+=1
    print "processing subject %s: %s " %(subject_num, img_file)

    try:    
        f = np.load(img_file)
        imgs_to_process=f['X'] 
        label=f['y']
    except:
        # store brocken npz into csv file
        with open(path2scorescsv, 'a') as f:
            string = str([subject_num,img_file])
            f.write(string + '\n')        
        continue            
        
    imgs_to_process= imgs_to_process.astype(np.float64)
    
    for i in range(len(imgs_to_process)):
        img = imgs_to_process[i]
        
        #Standardize the pixel values
        mean = np.mean(img)
        std = np.std(img)
        img = img-mean
        img = img/std
        
        # Find the average pixel value near the lungs
        # to renormalize washed out images
        middle = img[100:400,100:400] 
        mean = np.mean(middle)  
        max = np.max(img)
        min = np.min(img)
        
        # To improve threshold finding, I'm moving the 
        # underflow and overflow on the pixel spectrum
        img[img==max]=mean
        img[img==min]=mean
        #
        # Using Kmeans to separate foreground (radio-opaque tissue)
        # and background (radio transparent tissue ie lungs)
        # Doing this only on the center of the image to avoid 
        # the non-tissue parts of the image as much as possible
        #
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
        #
        # I found an initial erosion helful for removing graininess from some of the regions
        # and then large dialation is used to make the lung region 
        # engulf the vessels and incursions into the lung cavity by 
        # radio opaque tissue
        #
        eroded = morphology.erosion(thresh_img,np.ones([4,4]))
        dilation = morphology.dilation(eroded,np.ones([10,10]))
        #
        #  Label each region and obtain the region properties
        #  The background region is removed by removing regions 
        #  with a bbox that is to large in either dimnsion
        #  Also, the lungs are generally far away from the top 
        #  and bottom of the image, so any regions that are too
        #  close to the top and bottom are removed
        #  This does not produce a perfect segmentation of the lungs
        #  from the image, but it is surprisingly good considering its
        #  simplicity. 
        #
        labels = measure.label(dilation)
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
                good_labels.append(prop.label)
        mask = np.ndarray([512,512],dtype=np.int8)
        mask[:] = 0
        #
        #  The mask here is the mask for the lungs--not the nodes
        #  After just the lungs are left, we do another large dilation
        #  in order to fill in and out the lung mask 
        #
        for N in good_labels:
            mask = mask + np.where(labels==N,1,0)
        mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
        imgs_to_process[i] = mask
    
    Y=np.asarray(imgs_to_process,'uint8')
    #np.savez(img_file.replace(".npz","_lungmask"),Y=np.asarray(imgs_to_process,'uint8'))
    # Dump the created data to disk
    joblib.dump(Y,img_file.replace(".npz","_lungmask.joblib"))
    elapsed_time=(time.time()-start_time)
    
    print 'elapsed time: %.1f  sec' %(elapsed_time)
#%%
