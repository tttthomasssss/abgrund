from __future__ import division
__author__ = 'thomas'

import numpy as np


def logistic_sigmoid(X):
	return 1. / (1. + np.exp(-X))

# see http://feature-space.com/en/post24.html
# Recheck gradient of logistic sigmoid
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