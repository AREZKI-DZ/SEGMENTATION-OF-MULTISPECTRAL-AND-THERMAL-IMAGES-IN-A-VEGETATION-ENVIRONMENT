

# -*- coding:Utf_8 -*-

import numpy as np
import sys
import os

#In this module, we have created two functions: one for data standardization
# to avoid any anomaly on this data, and another funtionto measure prediction accuracy.

#Fonction of normalization of data by image
def Normalization(image):
	''' Function for the normalization of data by image'''
	image_norm=np.zeros_like(image)
	# The normalization of image
	for d in range(0,np.shape(image)[2]): 
		image_norm[:,:,d]=(image[:,:,d]-image[:,:,d].min())/(image[:,:,d].max()-image[:,:,d].min())*255
	image_norm=np.uint8(image_norm) 
	return image_norm


#The measure's function
def measure_accuracy(pred,mask_ground_truth):
	'''Function compute the precision of prediction of our model'''
	INPUT_WIDTH=np.shape(pred)[0]
	INPUT_HEIGHT=np.shape(pred)[1]
	size=INPUT_WIDTH*INPUT_HEIGHT
	nbGoodBlack=0
	nbGoodWhite=0
	for i in range(INPUT_WIDTH):
		for j in range(INPUT_HEIGHT):
			if int(pred[i,j])==int(0) and int(mask_ground_truth[i,j])==int(0):
				nbGoodBlack+=1

			elif int(pred[i,j])==int(255) and int(mask_ground_truth[i,j])==int(255):
				nbGoodWhite+=1

	return 100.0*(nbGoodBlack+nbGoodWhite)/size




