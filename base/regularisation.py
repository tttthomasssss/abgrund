__author__ = 'thomas'

from climin.util import shaped_from_flat
import numpy as np


def get_regularisation_fn_for_string(regularisation):
	if (regularisation == 'l2'):
		return l2_regularisation, deriv_l2_regularisation
	elif (regularisation == 'l1'):
		return l1_regularisation, deriv_l1_regularisation


def l2_regularisation(lambda_, W, shape):
	views = shaped_from_flat(W, shape)
	reg = 0
	for i in xrange(len(views)):
		if (views[i].ndim > 1): # Don't add regularisation for bias term
			reg += (views[i]**2).sum()

	return reg * lambda_


def deriv_l2_regularisation(lambda_, W):
	return W * lambda_


def l1_regularisation(lambda_, W, shape):
	views = shaped_from_flat(W, shape)
	reg = 0
	for i in xrange(len(views)):
		if (views[i].ndim > 1): # No regularisation for the biases
			reg += np.abs(views[i]).sum()

	return reg * lambda_


def deriv_l1_regularisation(lambda_, W, _):
	return (W > 0) * lambda_