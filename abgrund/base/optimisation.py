__author__ = 'thomas'


def gd(weights, gradients, **kwargs): # Gradient Descent (can handle SGD, Mini-Batch SGD and Batch GD)
	w_updated = []
	eta = kwargs.pop('eta', 0.01)
	for W, dg_dW in zip(weights, gradients):
		w_updated.append(W - (eta * dg_dW))

	return w_updated

def adagrad(weights, gradients, gradient_history, **kwargs):
	w_updated = []
	eta = kwargs.pop('eta', 0.01)


class AdaGrad():
	def __init__(self, eta=0.01):
		self.eta_ = eta
		self.gradient_history = []

	def __call__(self, *args, **kwargs):
		pass

# Vanilla SGD (+ with tricks),
# AdaGrad
# AdaDelta
# RMSProp (see hintons slide)
# Adam
# LBFGS

# http://sebastianruder.com/optimizing-gradient-descent/index.html#adagrad
# http://deliprao.com/archives/153
