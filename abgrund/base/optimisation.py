__author__ = 'thomas'
import numpy as np


class NoisyGradientDescent():
	def __init__(self, shape, eta=0.01, noise_mean=0, noise_eta=0.01, random_state=np.random.RandomState(1105), **_):
		self.shape_ = shape
		self.eta_ = eta
		self.noise_mean_ = noise_mean
		self.noise_eta_ = noise_eta
		self.random_state_ = random_state

	def __call__(self, *_, **kwargs):
		weights = kwargs['weights']
		gradients = kwargs['gradients']
		time_step = kwargs['time_step']

		sigma = self.noise_eta_ / ((1 + time_step) ** 0.55)

		w_updated = []

		for W, dg_dW, shape in zip(weights, gradients, self.shape_):
			w_updated.append(W - (self.eta_ * (dg_dW + self.random_state_.normal(self.noise_mean_, sigma, shape))))

		return w_updated


class GradientDescent(): # Gradient Descent (can handle SGD, Mini-Batch SGD and Batch GD)
	def __init__(self, shape, eta=0.01, momentum=None, v=0, mu=0.9, **_):
		self.eta_ = eta
		self.weight_update_fn_ = getattr(self, '_{}_momentum_weight_update'.format(momentum), self._vanilla_weight_update)
		self.shape_ = shape
		self.v_ = []
		self.mu_ = []

		if (momentum is not None):
			for s in self.shape_:
				self.v_.append(np.full(s, v))
				self.mu_.append(np.full(s, mu))

	def __call__(self, *_, **kwargs):
		weights = kwargs['weights']
		gradients = kwargs['gradients']

		w_updated = []

		for idx, (W, dg_dW) in enumerate(zip(weights, gradients)):
			w_updated.append(self.weight_update_fn_(W, dg_dW, idx))

		return w_updated

	def _vanilla_weight_update(self, W, dg_dW, _):
		return W - (self.eta_ * dg_dW)

	def _standard_momentum_weight_update(self, W, dg_dW, idx):
		self.v_ = (self.mu_[idx] * self.v_[idx]) - (self.eta_ * dg_dW)

		return W + self.v_

	def _nestorov_momentum_weight_update(self, W, dg_dW, idx):
		raise NotImplementedError #http://cs231n.github.io/neural-networks-3/#update


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


def gd(**kwargs):
	return GradientDescent(**kwargs)


def ngd(**kwargs):
	return NoisyGradientDescent(**kwargs)


def adagrad(**kwargs):
	return AdaGrad(**kwargs)

# Vanilla SGD (+ with tricks),
# AdaGrad
# AdaDelta
# RMSProp (see hintons slide)
# Adam
# LBFGS

# http://sebastianruder.com/optimizing-gradient-descent/index.html#adagrad
# http://deliprao.com/archives/153
