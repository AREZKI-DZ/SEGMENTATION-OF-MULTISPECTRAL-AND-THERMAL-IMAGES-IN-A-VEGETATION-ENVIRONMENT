
# -*- coding:Utf_8 -*-
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
import scipy.misc as im
from sklearn import svm
import seaborn as sns
from sklearn.externals import joblib
import numpy as np
import os
import sys

import measure_Functions

# This Function "Construct_X_Y" is to load the X and Y learning vectors for the model SVM, these vectors are made up of super-pixels.
# These super pixels can be defined as pixel regions that have the same characteristics. 
# Vector X is constructed from multispectral and thermal images, i.e. images of seven wavelengths,
# and the Y vector is constructed from the masks, in the same way.

def Construct_X_Y(path,n_segments, compactness):
    #X superpixel features or caracteristics
    #Y class identifier
	X=[]
	Y=[]
	#Load data
	for data in os.listdir(path+'/data'):
		if not data.find('Train') ==-1:
			pathSubset=path+'/data/'
			for image in os.listdir(pathSubset+data):
				pathArrayImgMask=pathSubset+data+'/'+image+'/ImageAndMaskArray'
				for image_mask in os.listdir(pathArrayImgMask):
					if image_mask.find('image')>1:
						image=np.load(pathArrayImgMask+'/'+image_mask).astype('double')
					else:
						mask=np.load(pathArrayImgMask+'/'+image_mask)
					
				#Normalization image
				image=measure_Functions.Normalization(image)
				
				#number of segmentations: n_segments
				# use the SLIC algorithm
				superpixel_labels = slic(image, n_segments=n_segments, compactness=compactness)

				# the channels of image
				B1 = image[:, :, 0]
				B2 = image[:, :, 1]
				B3 = image[:, :, 2]
				B4 = image[:, :, 3]
				B5 = image[:, :, 4]
				B6 = image[:, :, 5]
				B7 = image[:, :, 6]

				#The number of superpixels
				nb_superpixels = np.max(superpixel_labels) + 1
				#Means colors
				for label in range(nb_superpixels):
				    idx = superpixel_labels == label
				    B1[idx] = np.mean(B1[idx])
				    B2[idx] = np.mean(B2[idx])
				    B3[idx] = np.mean(B3[idx])
				    B4[idx] = np.mean(B4[idx])
				    B5[idx] = np.mean(B5[idx])
				    B6[idx] = np.mean(B6[idx])
				    B7[idx] = np.mean(B7[idx])

				# Borduries of each supepixel
				image_with_boundaries = mark_boundaries(image, superpixel_labels, \
				                      color=(1, 1, 1, 1, 1, 1, 1), outline_color=(0, 0, 0, 0, 0, 0, 0))
				# The parameters of image
				width = np.shape(B1)[1]
				height = np.shape(B1)[0]

				# pour calculer la position du barycentre dans la largeur de l'image
				x_idx = np.repeat(range(width), height)
				x_idx = np.reshape(x_idx, [width, height])
				x_idx = np.transpose(x_idx)

				# THE PARYCENTER OF EACH SUPER-PIXELS ON WEITH
				y_idx = np.repeat(range(height), width)
				y_idx = np.reshape(y_idx, [height, width])

				   
				    ######################################################################
				    ##########################.  TRAINING. ##############################@
				# THE EXTRACTION OF CARACTERISTICS FOR EACH SUPERPIXELS
				feature_superpixels = []
				for label in range(nb_superpixels): 
				    # THE PIXELS OF SUPER-PEXELS
				    idx = superpixel_labels == label
				    
				    # THE NORMALISATION FOR EACH CHANNEL
				    c1_mean = np.mean(B1[idx]-np.min(B1)) / (np.max(B1)-np.min(B1))
				    c2_mean = np.mean(B2[idx]-np.min(B2)) / (np.max(B2)-np.min(B2))
				    c3_mean = np.mean(B3[idx]-np.min(B3)) / (np.max(B3)-np.min(B3))
				    c4_mean = np.mean(B4[idx]--np.min(B4)) / (np.max(B4)-np.min(B4))
				    c5_mean = np.mean(B5[idx]--np.min(B5)) / (np.max(B5)-np.min(B5))
				    c6_mean = np.mean(B6[idx]--np.min(B6)) / (np.max(B6)-np.min(B6)) 
				    c7_mean = np.mean(B7[idx]--np.min(B7)) / (np.max(B7)-np.min(B7))
				    # calcul et normalisation de la position du barycentre
				    x_mean = np.mean(x_idx[idx]) / (width - 1)
				    y_mean = np.mean(y_idx[idx]) / (height - 1) 
				 
				    # THE VECTOR OF CARACTERISTICS
				    sp = [c1_mean, c2_mean, c3_mean,c4_mean,c5_mean,c6_mean,c7_mean, x_mean, y_mean]
				    feature_superpixels.append(sp)
				# recover the image color channels
				B1_learning = mask[:, :, 0]*255
				B2_learning = mask[:, :, 0]*255
				B3_learning = mask[:, :, 0]*255
				B4_learning = mask[:, :, 0]*255
				B5_learning = mask[:, :, 0]*255
				B6_learning = mask[:, :, 0]*255
				B7_learning = mask[:, :, 0]*255
				class_colors = [(B1_learning[y, x], B2_learning[y, x], B3_learning[y, x],B4_learning[y, x],B5_learning[y, x],B6_learning[y, x],B7_learning[y, x]) for y in range(height) \
				                for x in range(width)];class_colors = set(class_colors);class_colors = list(class_colors)
				np.save('class_colors.npy',class_colors)
				class_pixels = []
				for color in class_colors:
					learning_pixels = (B1_learning == color[0]) \
				                      & (B2_learning == color[1]) \
				                      & (B3_learning == color[2]) \
				                      & (B4_learning == color[3]) \
				                      & (B5_learning == color[4]) \
				                      & (B6_learning == color[5]) \
				                      & (B7_learning == color[6])
					class_pixels.append(learning_pixels)
				 
			  
				# recover some representative superpixels for each class
				for label in range(nb_superpixels):
                    # browse all the pixels of the
                    # superpixel and see how much
                    # of them are attributed to
                    # each class
				    nb_for_each_class = []
				    idx_sp = superpixel_labels == int(label)
				    for learning_pixels in class_pixels:
				        common_idx = np.logical_and(learning_pixels, idx_sp)
				        nb_for_each_class.append(np.sum(common_idx))
                    # test if the superpixel contains pixels
                    # belonging to one and only one class
				    class_idx = -1
				    several_classes = 0
				    for idx in range(len(nb_for_each_class)):
				        if nb_for_each_class[idx] > 0:
				            if class_idx < 0:
                                # the superpixel contains
                                # the pixels belonging
                                # to one of the classes
				                class_idx = idx
				            else:
                                # the superpixel contains
                                # of pixels owned
                                # to several classes:
                                # do not keep it as
                                # learning data
				                several_classes = True
                    # if the superpixel was retained as given
                    # learning, we store its characteristics
                    # and the class identifier
				    if (class_idx >= 0) and not several_classes:
				    	Y.append(class_idx)
				    	X.append(feature_superpixels[label])
                        #X, which contains the vectors describing the superpixels used during learning;
                        #Y, which associates each superpixel with a class.
	
	return X,Y, class_colors




