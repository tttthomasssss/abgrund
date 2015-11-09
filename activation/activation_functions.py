
__author__ = 'thomas'

import numpy as np

# TODO: Probably delete the whole shit
def logistic_sigmoid(X):
	return 1. / (1. + np.exp(-X))

# TODO: Recheck gradient of logistic sigmoid
# see http://feature-space.com/en/post24.html
# http://www.willamette.edu/~gorr/classes/cs449/Maple/ActivationFuncs/active.html
# http://www.ai.mit.edu/courses/6.892/lecture8-html/sld015.htm
# https://theclevermachine.wordpress.com/2014/09/08/derivation-derivatives-for-common-neural-network-activation-functions/
def derivative_logistic_sigmoid(X):
	g = logistic_sigmoid(X)

	return g * (1 - g)


def softplus(X):
	return X


def reLU(X):
	return np.maximum(X, 0) # <-- This is memory intensive!!! find a sparse way


def identity(X):
	return X


def heaviside(X, threshold=0.5):
	return X


def tanh_sigmoid(X):
	return X


def lecun_tanh_sigmoid(X): # LeCun (1998) Efficient Backprop
	return X