
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
import sys
from skimage.io import imsave,imread
from sklearn.externals import joblib


#load measurement function
import measure_Functions

# This function allows you to calculate the prediction accuracy, by comparing the 
# ground truth masks of field and the predicted masks, pixels-by-pixels.

def mesure_global(Path):
    
    Path=mainPath+'/data/Test_data/'
    glob_measure=0
    nb_subset=0
    for subset in os.listdir(Path):
        for mask in os.listdir(Path+subset+'/ImageAndMaskTif/'):
            if mask.find('pred')>1:
                image_pred=imread(Path+subset+'/ImageAndMaskTif/'+mask)
                print(subset)
                print(np.shape(image_pred))
                print(np.unique(image_pred))
            else:
                if mask.find('mask')>1:
                    Ground_truth=imread(Path+subset+'/ImageAndMaskTif/'+mask)
                    print(np.shape(Ground_truth))
                    print(np.unique(Ground_truth))

        measure=measure_Functions.measure_accuracy(image_pred,Ground_truth)
        print("the prediction's measure of "+str(subset)+" is :", measure )
        glob_measure+=measure
        nb_subset+=1
    print('The global measure is',glob_measure/nb_subset)

	return glob_measure/nb_subset
