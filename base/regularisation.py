__author__ = 'thomas'

from climin.util import shaped_from_flat
import numpy as np


def get_regularisation_fn_for_string(regularisation):
	if (regularisation == 'l2'):
		return l2_regularisation, deriv_l2_regularisation
	elif (regularisation == 'l1'):
		return l1_regularisation, deriv_l1_regularisation


def l2_regularisation(lambda_, W, shape, skip_first=False): # skip_first responsible for skipping the lookup layer
	views = shaped_from_flat(W, shape)
	views = views if not skip_first else views[1:]
	reg = 0
	while len(views) > 0:
		W_curr = views.pop()
		if (W_curr.ndim > 1): # Don't add regularisation for bias term
			reg += (W_curr**2).sum()

	return reg * lambda_


def deriv_l2_regularisation(lambda_, W):
	return W * lambda_


def l1_regularisation(lambda_, W, shape, skip_first=False): # skip_first responsible for skipping the lookup layer
	views = shaped_from_flat(W, shape)
	views = views if not skip_first else views[1:]
	reg = 0
	for i in range(len(views)):
		if (views[i].ndim > 1): # No regularisation for the biases
			reg += np.abs(views[i]).sum()

	return reg * lambda_


def deriv_l1_regularisation(lambda_, W, _):
	return (W > 0) * lambda_