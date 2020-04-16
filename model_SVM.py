
# -*- coding:Utf_8 -*-
from sklearn import svm
import seaborn as sns
import numpy as np
import os
import sys
from sklearn.externals import joblib


def Train(X,Y):

	# creer le séparateur à vaste marge
	model_svm = svm.SVC(decision_function_shape='ovo')
	# paramètre du SVM permettant d'influencer la proportion
	# de données d'apprentissage pouvant être considérées comme
	# erronées
	model_svm.C = 4.
	# paramètre du noyau du SVM 
	model_svm.gamma = 4.
	# indiquer que les probabilités d'appartenir à chaque classe 
	# doivent être calculées
	model_svm.probability = True
	# entraîner le SVM 
	model_svm.fit(X,Y)
	#Classification and prediction
	# save the model to disk
	joblib.dump(model_svm,'model_SVM.sav')

	return model_svm

