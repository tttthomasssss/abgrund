__author__ = 'thomas'

import numpy as np


def l2_regularisation(lambda_, weights, skip_first=False): # skip_first responsible for skipping the lookup layer
	skip = 0 if not skip_first else 1
	reg = 0
	for W in weights[skip:]:
		if (W.ndim > 1): # Don't add regularisation for bias term
			reg += (W**2).sum()

	return reg * lambda_


def deriv_l2_regularisation(lambda_, W):
	return W * lambda_


def l1_regularisation(lambda_, weights, skip_first=False): # skip_first responsible for skipping the lookup layer
	skip = 0 if not skip_first else 1
	reg = 0
	for W in weights[skip:]:
		if (W.ndim > 1): # No regularisation for the biases
			reg += np.abs(W).sum()

	return reg * lambda_


def deriv_l1_regularisation(lambda_, W, _):
	return (W > 0) * lambda_