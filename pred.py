
# -*- coding:Utf_8 -*-
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import scipy.misc as im
from sklearn import svm
import seaborn as sns
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from skimage.io import imsave,imread
from sklearn.externals import joblib


#Normalization data
import measure_Functions

# This function 'Pred_Data' is for use the model trained on the learning image. 
# It can predict the masks of the test images and then record them in the same folders as those images.

def Pred_Data(mainPath, model_svm,n_segments=2000, compactness=10, class_colors=[]):
    for data in os.listdir(mainPath+'/data'):
        if data.find('Train') ==-1:
            pathSubset=mainPath+'/data/'
            for subset in os.listdir(pathSubset+data):
                pathArrayImgMask=pathSubset+data+'/'+subset+'/ImageAndMaskArray'
                for imag_to_pred in os.listdir(pathArrayImgMask):
                    if imag_to_pred.find('image')>1:
                        image=np.load(pathArrayImgMask+'/'+imag_to_pred)
                        
                # The normalization of data
                # The normalization of image
                image=measure_Functions.Normalization(image)
                #number of segmentations: n_segments
                # use the SLIC algorithm
                superpixel_labels = slic(image, n_segments=2000, compactness=10)
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

                # calculate the position of the barycenter in the width of the image
                x_idx = np.repeat(range(width), height)
                x_idx = np.reshape(x_idx, [width, height])
                x_idx = np.transpose(x_idx)

                # THE PARYCENTER OF EACH SUPER-PIXELS ON WEITH
                y_idx = np.repeat(range(height), width)
                y_idx = np.reshape(y_idx, [height, width])

                # extracting the characteristics of each channel
                feature_superpixels = []
                for label in range(nb_superpixels): 
                    # THE PIXELS OF SUPER-PEXELS
                    idx = superpixel_labels == label

                    # the barycenter of each channel
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

                    # The barycenter vector
                    sp = [c1_mean, c2_mean, c3_mean,c4_mean,c5_mean,c6_mean,c7_mean, x_mean, y_mean]
                    feature_superpixels.append(sp)

                # predict the probability of each superpixel
                # to belong to each class
                # We have two classes  
                probas = model_svm.predict_proba(feature_superpixels)
                # predire la classe la plus probable pour chaque superpixel
                classification = model_svm.predict(feature_superpixels)
                # parcourir chacune des classes
                for class_id in range(len(class_colors)):
                    pixel_probas = np.zeros([height, width])
                    # transfer the probability of the superpixel
                    # to the pixels that constitute it
                    for label in range(nb_superpixels):
                        idx = superpixel_labels == label
                        pixel_probas[idx] = probas[label, class_id]
                    # show result
                    for i in range(0,np.shape(image)[0]): 
                        for j in range(0,np.shape(image)[1]):
                            if pixel_probas[i,j] > 0.5: 
                                pixel_probas[i,j]=1 
                            else: 
                                pixel_probas[i,j]=0
                    imsave(pathSubset+data+'/'+subset+'/ImageAndMaskTif/'+str(imag_to_pred[0:len(imag_to_pred)-4])+'_pred.tiff',np.uint8(pixel_probas*255))
