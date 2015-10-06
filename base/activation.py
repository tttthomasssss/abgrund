__author__ = 'thomas'
import numpy as np


def get_activation_fn_for_string(activation):
	if (activation == 'sigmoid'):
		return (sigmoid, deriv_sigmoid)
	elif (activation == 'relu'):
		return (relu, deriv_relu)
	elif (activation == 'tanh'):
		return (tanh, deriv_tanh)


def get_prediction_fn_for_string(prediction):
	if (prediction == 'softmax'):
		return (softmax, deriv_softmax)


def softmax(x):
	softmaxed = np.exp(x - x.max(axis=1)[:, np.newaxis]) # Numerical Stability!

	return softmaxed / softmaxed.sum(axis=1)[:, np.newaxis]


def deriv_softmax(y_pred, y_true):
	return y_pred - y_true


def sigmoid(x):
	g = np.zeros(x.shape) # Numerical Stability, see: http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
	g[x >= 0] = 1 / (1 + np.exp(-x[x >= 0]))
	g[x < 0] = 1 / (1 + np.exp(x[x < 0]))
	return g


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