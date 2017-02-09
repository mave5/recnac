import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
from sklearn.externals import joblib
import cv2
import matplotlib.pylab as plt
from sklearn import cross_validation
#%%
working_path = "./output/numpy/luna/allsubsets/"

file_list=glob(working_path+"images_*.npy")
print 'total images %s' %(len(file_list))

mask_list=glob(working_path+"masks_*.npy")
print 'total images %s' %(len(mask_list))

lungmask_list=glob(working_path+"lungmask_*.npy")
print 'total images %s' %(len(lungmask_list))
#%%

def grays_to_RGB(img):
    # turn 2D grayscale image into grayscale RGB
    return np.dstack((img, img, img))


def image_with_mask(img, mask,color=(0,255,0)):
    maximg=np.max(img)    
    img=np.asarray(img,dtype='float32')
    img=np.asarray((img/maximg)*255,dtype='uint8')
    mask=np.asarray(mask,dtype='uint8') 
    if np.max(mask)==1:
        mask=mask*255

    # returns a copy of the image with edges of the mask added in red
    if len(img.shape)==2:	
	img_color = grays_to_RGB(img)
    else:
	img_color =img

    mask_edges = cv2.Canny(mask, 100, 200) > 0
    img_color[mask_edges, 0] = color[0]  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask_edges, 1] = color[1]
    img_color[mask_edges, 2] = color[2]
    img_color=img_color#/float(np.max(img))
    return img_color

def disp_img_2masks(img,mask1,mask2,r=1,c=1,d=0,indices=None):
    if mask1 is None:
        mask1=np.zeros(img.shape,dtype='uint8')
    if mask2 is None:
        mask2=np.zeros(img.shape,dtype='uint8')
        
    N=r*c    
    if d==2:
        # convert to N*C*H*W
        img=np.transpose(img,(2,0,1))
        img=np.expand_dims(img,axis=1)
        
        mask1=np.transpose(mask1,(2,0,1))
        mask1=np.expand_dims(mask1,axis=1)

        mask2=np.transpose(mask2,(2,0,1))
        mask2=np.expand_dims(mask2,axis=1)
        
    if indices is None:    
        # random indices   
        n1=np.random.randint(img.shape[0],size=N)
    else:
        n1=indices
    
    I1=img[n1,0]
    #M1=mask1[n1,0]
    M1=np.zeros(I1.shape,dtype='uint8')
    for c1 in range(mask1.shape[1]):
        M1=np.logical_or(M1,mask1[n1,c1,:])    
    #M2=mask2[n1,0]
    M2=np.zeros(I1.shape,dtype='uint8')
    for c1 in range(mask2.shape[1]):
        M2=np.logical_or(M2,mask2[n1,c1,:])    
    
    C1=(0,255,9)
    C2=(255,0,0)
    for k in range(N):    
        imgmask=image_with_mask(I1[k],M1[k],C1)
        imgmask=image_with_mask(imgmask,M2[k],C2)
        plt.subplot(r,c,k+1)
        plt.imshow(imgmask)
        plt.title(n1[k])
    plt.show()            
    return n1        

# train data collection

def stack_data(trn_fl):
    out_images = []      #final set of images
    out_nodemasks = []   #final set of nodemasks
    
    for sbnb,fname in enumerate(trn_fl):
        print "working on file: %s, %s" %(sbnb,fname)
        
        imgs_to_process = np.load(fname.replace("lungmask","images"))
        masks = np.load(fname)
        node_masks = np.load(fname.replace("lungmask","masks"))
        
        for i in range(len(imgs_to_process)):
            mask = masks[i]
            node_mask = node_masks[i]
            img = imgs_to_process[i]
            #new_size = [512,512]   # we're scaling back up to the original size of the image
            img= mask*img          # apply lung mask
            #
            # renormalizing the masked image (in the mask region)
            #
            new_mean = np.mean(img[mask>0])  
            new_std = np.std(img[mask>0])
            #
            #  Pulling the background color up to the lower end
            #  of the pixel range for the lungs
            #
            old_min = np.min(img)       # background color
            img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
            img = img-new_mean
            img = img/new_std
            #make image bounding box  (min row, min col, max row, max col)
            labels = measure.label(mask)
            regions = measure.regionprops(labels)
            #
            # Finding the global min and max row over all regions
            #
            min_row = 512
            max_row = 0
            min_col = 512
            max_col = 0
            for prop in regions:
                B = prop.bbox
                if min_row > B[0]:
                    min_row = B[0]
                if min_col > B[1]:
                    min_col = B[1]
                if max_row < B[2]:
                    max_row = B[2]
                if max_col < B[3]:
                    max_col = B[3]
            width = max_col-min_col
            height = max_row - min_row
            if width > height:
                max_row=min_row+width
            else:
                max_col = min_col+height
            # 
            # cropping the image down to the bounding box for all regions
            # (there's probably an skimage command that can do this in one line)
            # 
            img = img[min_row:max_row,min_col:max_col]
            mask =  mask[min_row:max_row,min_col:max_col]
            if max_row-min_row <5 or max_col-min_col<5:  # skipping all images with no god regions
                pass
            else:
                # moving range to -1 to 1 to accomodate the resize function
                mean = np.mean(img)
                img = img - mean
                min = np.min(img)
                max = np.max(img)
                img = img/(max-min)
                new_img = resize(img,[512,512])
                new_node_mask = resize(255*node_mask[min_row:max_row,min_col:max_col],[512,512]).astype(np.uint8)
                out_images.append(new_img)
                out_nodemasks.append(new_node_mask)
    
    num_images = len(out_images)
    #
    #  Writing out images and masks as 1 channel arrays for input into network
    #
    final_images = np.ndarray([num_images,1,512,512],dtype=np.float32)
    final_masks = np.ndarray([num_images,1,512,512],dtype=np.uint8)
    for i in range(num_images):
        final_images[i,0] = out_images[i]
        final_masks[i,0] = out_nodemasks[i]
    
    return final_images,final_masks



#%%
for img_file in file_list[636:]:
    # I ran into an error when using Kmean on np.float16, so I'm using np.float64 here
    print "processing subject:", img_file
    imgs_to_process = np.load(img_file).astype(np.float64) 
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
    np.save(img_file.replace("images","lungmask"),imgs_to_process)
    
#%%
#
#    Here we're applying the masks and cropping and resizing the image
#
file_list=glob(working_path+"lungmask_*.npy")
print 'total lung masks %s' %(len(lungmask_list))



trn_fl, val_fl, _, _ = cross_validation.train_test_split(file_list, file_list, random_state=420, 
                                                                   test_size=0.1)
print 'number of train masks %s' %(len(trn_fl))
print 'number of val masks %s' %(len(val_fl))

#%% stack train data
X_train,Y_train=stack_data(trn_fl)    
joblib.dump(X_train,working_path+"trainImages.joblib")
joblib.dump(Y_train,working_path+"trainMasks.joblib")

# stack test data
X_test,Y_test=stack_data(val_fl)    
joblib.dump(X_test,working_path+"testImages.joblib")
joblib.dump(Y_test,working_path+"testMasks.joblib")   

#%% verify

prefix='test'
X=joblib.load(working_path+prefix+"Images.joblib")
print X.shape

Y=joblib.load(working_path+prefix+"Masks.joblib")
print Y.shape

plt.figure(figsize=(20,100))
n1=disp_img_2masks(X,Y,None,4,5,True)


