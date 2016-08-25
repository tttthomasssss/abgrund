__author__ = 'thomas'
import numpy as np


class GradientDescent(): # Gradient Descent (can handle SGD, Mini-Batch SGD and Batch GD)
	def __init__(self, eta, **_):
		self.eta_ = eta

	def __call__(self, *_, **kwargs):
		weights = kwargs['weights']
		gradients = kwargs['gradients']

		w_updated = []

		for W, dg_dW in zip(weights, gradients):
			w_updated.append(W - (self.eta_ * dg_dW))

		return w_updated


def gd(**kwargs):
	return GradientDescent(**kwargs)

def adagrad(**kwargs):
	return AdaGrad(**kwargs)

#def gd(weights, gradients, **kwargs): # Gradient Descent (can handle SGD, Mini-Batch SGD and Batch GD)
#	w_updated = []
#	eta = kwargs.pop('eta', 0.01)
#	for W, dg_dW in zip(weights, gradients):
#		w_updated.append(W - (eta * dg_dW))
#
#	return w_updated


class AdaGrad():
	def __init__(self, shape, eta=0.01, eps=1e-8, **_):
		self.eta_ = eta
		self.eps_ = eps
		self.gradient_history = []

		for s in shape:
			self.gradient_history.append(np.zeros(s)) # Initialise Gradient History

	def __call__(self, *_, **kwargs):
		weights = kwargs['weights']
		gradients = kwargs['gradients']

		w_updated = []

		for i, (W, dg_dW) in enumerate(zip(weights, gradients)):
			# Update Gradient History
			self.gradient_history[i] += dg_dW ** 2

			# Calculate AdaGrad magic term
			ada = self.eta_ / (np.sqrt(self.gradient_history[i] + self.eps_))

			# Update weights
			w_updated.append(W - (ada * dg_dW))

		return w_updated

# Vanilla SGD (+ with tricks),
# AdaGrad
# AdaDelta
# RMSProp (see hintons slide)
# Adam
# LBFGS

# http://sebastianruder.com/optimizing-gradient-descent/index.html#adagrad
# http://deliprao.com/archives/153
