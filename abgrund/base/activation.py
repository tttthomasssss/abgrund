__author__ = 'thomas'
from scipy.special import expit
import numpy as np


def softmax(x):
	softmaxed = np.exp(x - x.max(axis=1)[:, np.newaxis]) # Numerical Stability!

	return softmaxed / softmaxed.sum(axis=1)[:, np.newaxis]


def deriv_softmax(y_pred, y_true):
	return y_pred - y_true


def sigmoid(x):
	return expit(x)


def deriv_sigmoid(x):
	sigm = sigmoid(x)

	return sigm * (1 - sigm)


def tanh(x):
	return np.tanh(x)


def deriv_tanh(x):
	return 1. - np.tanh(x)**2


def relu(x):
	return np.maximum(0, x)


def deriv_relu(x):
	return np.float64(x > 0)


def elu(x, alpha=1.): # Eqn. 16 - http://arxiv.org/pdf/1511.07289v1.pdf
	# alternatively but a little slower: (x >= 0) * x + (x < 0) * (alpha * (np.exp(x) - 1))
	return np.maximum(0, x) + alpha * (np.exp(np.minimum(0, x)) - 1)


def deriv_elu(x, alpha=1.): # Eqn. 17 - http://arxiv.org/pdf/1511.07289v1.pdf
	return np.float64(x >= 0) + (x < 0) * (alpha * (np.exp(x) - 1) + alpha)