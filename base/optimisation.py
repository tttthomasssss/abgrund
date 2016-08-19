__author__ = 'thomas'


def gd(weights, gradients, **kwargs): # Gradient Descent (can handle SGD, Mini-Batch SGD and Batch GD)
	w_updated = []
	eta = kwargs.pop('eta', 0.01)
	for W, dg_dW in zip(weights, gradients):
		w_updated.append(W - (eta * dg_dW))

	return w_updated
# Vanilla SGD (+ with tricks),
# AdaGrad
# AdaDelta
# RMSProp (see hintons slide)
# Adam
# LBFGS