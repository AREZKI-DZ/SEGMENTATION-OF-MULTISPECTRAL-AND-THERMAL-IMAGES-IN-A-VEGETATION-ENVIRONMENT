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


#importing models
import model_SVM
import train_SVM
import pred
import measure_Functions


# This script is for loading learning data, learning the SVM model, 
# and then measuring the accuracy of the trained model SMV by making
# the prediction on test images. At the end, we will have an overall accuracy value of the trained model.

#Path of data training
mainPath=os.getcwd()

#Construction des vecteurs X, Y d'apprentissage
X,Y,class_colors=train_SVM.Construct_X_Y(mainPath,n_segments=4000, compactness=100)

#Training model
model_svm=model_SVM.Train(X,Y)


#Part of prediction 
pred.Pred_Data(mainPath, model_svm,n_segments=4000, compactness=100, class_colors=class_colors)

