__author__ = 'thomas'
import collections
import copy
import itertools
import json
import os
import sys

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import safe_sparse_dot
import numpy as np
import scipy as sp

from abgrund.base import activation
from abgrund.base import constraints
from abgrund.base import initialisation
from abgrund.base import optimisation
from abgrund.base import regularisation as reg
from abgrund.base import utils


# TODO: Early stopping when validation error doesnt go down for a number of specified epochs
#		Write Params to disk / Read params from disk
#		Not yet clear if Dropout is implemented correctly
#		Backprop into input vectors!!!
#		Re-Test Gradient check
class MLP(BaseEstimator):
	def __init__(self, shape, activation_fn='tanh', prediction_fn='softmax', w_init='xavier', gradient_check=False,
				 regularisation='l2', lambda_=0.01, dropout_proba=None, random_state=np.random.RandomState(seed=1105),
				 max_epochs=20, improvement_threshold=0.995, patience=np.inf, validation_frequency=100,
				 max_weight_norm=None, mini_batch_size=50, optimiser='gd', optimiser_kwargs={}):
		self.random_state_ = utils.create_random_state(random_state)
		self.shape_ = shape
		self.weights_ = self._initialise_weights(w_init, activation_fn)
		self.activation_fn, self.deriv_activation_fn = (getattr(activation, activation_fn), getattr(activation, 'deriv_{}'.format(activation_fn)))
		self.prediction_fn, self.deriv_prediction_fn = (getattr(activation, prediction_fn), getattr(activation, 'deriv_{}'.format(prediction_fn)))
		self.optimiser_ = getattr(optimisation, optimiser)
		self.optimiser_kwargs_ = optimiser_kwargs
		self.gradient_check_ = gradient_check
		self.lambda_ = lambda_
		self.regularisation_, self.deriv_regularisation_ = (getattr(reg, '{}_regularisation'.format(regularisation)), getattr(reg, 'deriv_{}_regularisation'.format(regularisation)))
		self.max_weight_norm_ = max_weight_norm
		self.dropout_proba_ = self._create_dropout_prediction_proba(dropout_proba)
		self.dropout_masks_ = self._create_dropout_masks(dropout_proba) # Be careful with dropout_proba, this fn modifies it
		self.max_epochs_ = np.inf if max_epochs is None or max_epochs < 0 else max_epochs
		self.improvement_threshold_ = improvement_threshold
		self.patience_ = patience
		self.best_weights_ = []
		self.validation_frequency_ = validation_frequency
		self.loss_history_ = []
		self.mini_batch_size_ = mini_batch_size
		self.num_classes_ = 0
		self.num_instances_ = 0

	def best_weights(self, flat=False):
		return self.best_weights_

	# Forward Propagation phase
	def _forward_propagation(self, X, dropout_mode='fit'):
		activations = [X]
		magnitudes = []
		dropout_iterator = iter(self.dropout_masks_) if dropout_mode == 'fit' else iter(self.dropout_proba_)

		idx = list(range(len(self.weights_)))
		b_pred = self.weights_[idx.pop()]
		W_pred = self.weights_[idx.pop()]

		# Forward Propagation through hidden layers
		while len(idx) > 1:
			W = self.weights_[idx.pop(0)]
			b = self.weights_[idx.pop(0)]

			# Dropout
			dropout_mask = next(dropout_iterator) if (len(self.dropout_masks_) > 0) else None
			W = self._perform_dropout(W, dropout_mode, dropout_mask)

			# Linear Transformation
			z = safe_sparse_dot(activations[-1], W) + b

			# Apply Non-Linearity
			a = self.activation_fn(z)

			activations.append(a)
			magnitudes.append(z)

		# Prediction - Dropout
		dropout_mask = next(dropout_iterator) if (len(self.dropout_masks_) > 0) else None
		W_pred = self._perform_dropout(W_pred, dropout_mode, dropout_mask)

		# Prediction - Linear Transformation
		z = safe_sparse_dot(activations[-1], W_pred) + b_pred

		# Predict!
		a = self.prediction_fn(z)

		activations.append(a)
		magnitudes.append(z)

		return activations, magnitudes

	def predict_proba(self, X):
		activations, _ = self._forward_propagation(X, dropout_mode='predict')

		return activations[-1]

	def predict(self, X):
		return np.argmax(self.predict_proba(X), axis=1)

	def loss(self, X, y):
		activations, _ = self._forward_propagation(X,dropout_mode='predict')
		predictions = activations[-1]

		loss = (np.nan_to_num(-np.log(predictions)) * utils.one_hot(y, s=self.num_classes_)).sum(axis=1).mean() # use np.nansum for this utils.one_hot(y)).sum(axis=1).mean()

		# Add regularisation
		reg = self.regularisation_(self.lambda_, self.weights_)
		loss += (reg / (self.num_instances_ * 2))

		return loss

	def fit(self, X, y, X_valid, y_valid):
		# Some initial administrative stuff
		self.num_classes_ = np.unique(y).shape[0]
		self.num_instances_ = utils.num_instances(X)
		Y = utils.one_hot(y, self.num_classes_)

		# Build index cycles over input data
		idx = np.arange(X.shape[0])
		if (self.mini_batch_size_ is not None and self.mini_batch_size_ > 0):
			idx_stream = itertools.repeat(np.array_split(idx, idx.shape[0] / self.mini_batch_size_), self.max_epochs_)
		else:
			idx_stream = itertools.repeat(np.array_split(idx, 1), self.max_epochs_) # Batched Input

		# Log initial performance
		y_pred = self.predict(X)
		print('Initial Training Loss={}; Training Accuracy={}'.format(self.loss(X, y), accuracy_score(y, y_pred)))
		if (X_valid is not None):
			y_pred = self.predict(X_valid)
			print('Initial Validation Loss={}; Validation Accuracy={}'.format(self.loss(X_valid, y_valid), accuracy_score(y_valid, y_pred)))
		print('----------------------------------------')

		# Run optimisation
		for epoch, idx_chunk in enumerate(idx_stream, 1): # Epoch cycle
			for mini_batch in idx_chunk: # Mini-Batch cycle
				gradients = self._backprop(X[mini_batch], Y[mini_batch])

				self.weights_ = self.optimiser_(self.weights_, gradients, **self.optimiser_kwargs_)

				# Max Norm constraint, see Hinton (2012) or Kim (2014) - often used in conjunction with dropout
				if (self.max_weight_norm_ is not None):
					self.weights_ = constraints.max_weight_norm(weights=self.weights_, max_norm=self.max_weight_norm_)

			# Log performance
			y_pred = self.predict(X)
			print('Training Loss={}; Accuracy={} after epoch {}'.format(self.loss(X, y), accuracy_score(y, y_pred), epoch))
			if (X_valid is not None):
				y_pred = self.predict(X_valid)
				print('Validation Loss={}; Validation Accuracy={} after epoch {}'.format(self.loss(X_valid, y_valid), accuracy_score(y_valid, y_pred), epoch))
			print('----------------------------------------')

	def _stopping_criterion(self, curr_iter, curr_patience, loss):
		if (np.isinf(curr_patience)):
			return curr_iter > self.max_epochs_ or loss <= 0
		else:
			return curr_patience <= 0 or loss <= 0

	def _gradient_check(self, W, X, y, eps=10e-4, error_threshold=10e-2):
		dg_dW = self._backprop(W, X, utils.one_hot(y, self.num_classes_))

		num_dg_dW = np.zeros(W.shape)
		perturb = np.zeros(W.shape)

		for i in range(W.shape[0]):
			perturb[i] = eps

			loss_plus = self.loss(W + perturb, X, y)
			loss_minus = self.loss(W - perturb, X, y)

			num_dg_dW[i] = (loss_plus - loss_minus) / (2 * eps)
			perturb[i] = 0

		diff = sp.linalg.norm(num_dg_dW - dg_dW) / sp.linalg.norm(num_dg_dW + dg_dW)

		return (diff <= error_threshold, diff, error_threshold)

	def _initialise_weights(self, W_init, activation_fn):
		if (W_init == 'xavier' and hasattr(initialisation, 'randomise_uniform_{}'.format(activation_fn))):
			return getattr(initialisation, 'randomise_uniform_{}'.format(activation_fn))(self.shape_)
		else:
			return initialisation.randn(self.shape_)

	def _perform_dropout(self, W, dropout_mode, dropout_mask):
		if (dropout_mask is not None):
			if (dropout_mode == 'fit'):
				W = W * dropout_mask[:, np.newaxis]
			else:
				W = W * dropout_mask # In that case its a probability

		return W

	def _create_dropout_prediction_proba(self, dropout_proba):
		l = []
		if (dropout_proba is not None):
			for p in dropout_proba:
				p_dropout = 1. - p if p is not None else 1.
				l.append(p_dropout)

		return l

	def _create_dropout_masks(self, dropout_proba):
		dropout_masks = []
		if (dropout_proba is not None):
			if (isinstance(dropout_proba, float)):
				for i in range(0, len(self.shape_) - 1, 2):
					size = (self.shape_[i][0],)
					dropout_masks.append(self.random_state_.binomial(1, dropout_proba, size))
			else:
				for i in range(0, len(self.shape_) - 1, 2):
					p = dropout_proba.pop(0)
					if (p is not None):
						size = (self.shape_[i][0],)
						dropout_masks.append(self.random_state_.binomial(1, p, size))
					else:
						dropout_masks.append(None)

		return dropout_masks

	def _backprop(self, X, y):
		# Backprop implemented by following http://neuralnetworksanddeeplearning.com/chap2.html
		gradients = []

		# Dropout Gradients
		if (self.dropout_masks_ is not None and len(self.dropout_masks_) > 0):
			gradients_dropout_iterator = reversed(self.dropout_masks_)

		# Prediction of network w.r.t. to current W
		activations, magnitudes = self._forward_propagation(X)

		# Pop weights and bias for last layer
		W_ = self.weights_[-2]

		# Pop activations and activation magnitudes
		y_pred = activations.pop() # Thats the prediction
		z = magnitudes.pop()

		# BP 1: Error in last layer
		delta_l = self.deriv_prediction_fn(y_pred, y) * self.deriv_activation_fn(z)

		# Pop another activation
		a = activations.pop()

		# Gradients w.r.t. last layer error
		de_dW = safe_sparse_dot(a.T, delta_l) / utils.num_instances(a) # BP 4: dot product between inputs that caused the error and backpropped error
		db_dW = delta_l.mean(axis=0) # BP 3: gradient of bias = delta_l

		# Add Gradient from Regularisation parameter
		de_dW += (self.deriv_regularisation_(self.lambda_, W_) / utils.num_instances(a))

		# Dropout during Backprop a.k.a. Backpropout
		if (self.dropout_masks_ is not None and len(self.dropout_masks_) > 0):
			gradients_dropout_mask = next(gradients_dropout_iterator)
			if (gradients_dropout_mask is not None):
				de_dW *= gradients_dropout_mask[:, np.newaxis]

		# Collect Gradients from last layer
		gradients.extend([de_dW, db_dW])

		# Loop through hidden layers
		i = 0 # Index on the weights
		while len(activations) > 0:
			W_next = self.weights_[-2-i] # Weights at lower (=next) layer
			W_curr = self.weights_[-2-i-2] # Weights at current layer

			a = activations.pop()
			z = magnitudes.pop()

			# BP 2: delta^(l) at hidden layer: backprop error signal
			delta_l = safe_sparse_dot(delta_l, W_next.T) * self.deriv_activation_fn(z)

			# BP 4: Gradients for weights w.r.t. backpropped error (delta_l) and forwardpropped activation
			de_dW = safe_sparse_dot(a.T, delta_l) / utils.num_instances(a)
			db_dW = delta_l.mean(axis=0)

			de_dW += (self.deriv_regularisation_(self.lambda_, W_curr) / utils.num_instances(a))

			# Dropout during Backprop a.k.a. Backpropout
			if (self.dropout_masks_ is not None and len(self.dropout_masks_) > 0):
				gradients_dropout_mask = next(gradients_dropout_iterator)
				if (gradients_dropout_mask is not None):
					de_dW *= gradients_dropout_mask[:, np.newaxis]

			# Collect Gradients
			gradients.insert(0, db_dW)
			gradients.insert(0, de_dW)

			# Update index into weights
			i += 2

		return gradients

if (__name__ == '__main__'):
	result_dict = {}
	#wrt = np.empty(159010)
	#initialize.randomize_normal(wrt, 0, 1)

	# Infer num params
	#shapes = [(i,) if isinstance(i, int) else i for i in shapes]
	#sizes = [np.prod(i) for i in shapes]

	#views = shaped_from_flat(wrt, tmpl)
	#wrt = initialize.randomize_uniform_sigmoid(views)

	### 20 NEWS GROUPS TEST WITH EXACTLY THE SAME PARAMS AS MNIST
	dataset = dataset_utils.fetch_20newsgroups_dataset_vectorized(os.path.join(paths.get_dataset_path(), '20newsgroups'), tf_normalisation=True)
	#X_train, y_train, X_test, y_test, Z = split_data(dataset, 0, 1, 2, 3, -1, np.random.RandomState(seed=42))
	X_train, y_train, X_valid, y_valid, X_test, y_test = split_data_train_dev_test('20newsgroups', dataset, ratio=(0.8, 0.2), random_state=np.random.RandomState(seed=42))

	######## S K L E A R N   C L A S S I F I E R S
	print('#### SKLEARN SVM')
	svm = LinearSVC()
	svm.fit(X_train, y_train)
	y_pred = svm.predict(X_test)
	print('\tAccuracy: {}; F1-Score: {}'.format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted' if len(np.unique(y_test)) > 2 else 'binary')))
	print('################')
	result_dict['svm_accuracy'] = accuracy_score(y_test, y_pred)
	result_dict['svm_f1_score'] = f1_score(y_test, y_pred, average='weighted' if len(np.unique(y_test)) > 2 else 'binary')

	print('#### SKLEARN MNB')
	mnb = MultinomialNB()
	mnb.fit(X_train, y_train)
	y_pred = mnb.predict(X_test)
	print('\tAccuracy: {}; F1-Score: {}'.format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted' if len(np.unique(y_test)) > 2 else 'binary')))
	print ('################')
	result_dict['nb_accuracy'] = accuracy_score(y_test, y_pred)
	result_dict['nb_f1_score'] = f1_score(y_test, y_pred, average='weighted' if len(np.unique(y_test)) > 2 else 'binary')
	##############################################

	timestamped_foldername = path_utils.timestamped_foldername()
	# TODO: Reproduce this paper: http://www.aclweb.org/anthology/P/P15/P15-1162.pdf
	act_fn = ['relu', 'tanh']#, 'relu', 'sigmoid']
	for afn in act_fn:

		print('#### {} ####'.format(afn))

		if (afn == 'relu'):
			gd_params = {'step_rate': 0.005, 'momentum': 0.95, 'momentum_type': 'nesterov'}
			optimiser_kwargs = {'eta': 4.}
		else:
			optimiser_kwargs = {'eta': 0.5}
			gd_params = {'step_rate': 0.1, 'momentum': 0.95, 'momentum_type': 'nesterov'}

		lbfgs_params = {'n_factors': 10}
		rmsprop_params = {'step_rate':0.01, 'decay':0.9, 'momentum':0, 'step_adapt':False, 'step_rate_min':0, 'step_rate_max':np.inf}
		adadelta_params = {'step_rate':0.05, 'decay':0.9, 'momentum':0, 'offset':1e-4}
		adam_params = {'step_rate':0.1, 'decay':1-1e-8, 'decay_mom1':0.1, 'decay_mom2':0.001, 'momentum':0, 'offset':1e-8}
		rprop_params = {'step_shrink':0.5, 'step_grow':1.2, 'min_step':1e-6, 'max_step':1, 'changes_max':0.1}
		nonlinearcg_params = {'min_grad':1e-6}

		#mlp = MLP(shape=[(8713, 500), 500, (500, 2), 2], dropout_proba=[None, 0.5, None], activation_fn=afn, improvement_threshold=0.995, patience=10, validation_frequency=100, optimiser='gd', **gd_params)
		#mlp = MLP(shape=[(130107, 500), 500, (500, 20), 20], gradient_check=True, dropout_proba=[None, 0.5, None], activation_fn=afn, max_epochs=200, validation_frequency=10, optimiser='gd', **gd_params)

		#mlp = MLP(shape=[(130107, 1500), 1500, (1500, 300), 300, (300, 20), 20], dropout_proba=None, activation_fn=afn, max_epochs=100, validation_frequency=10, optimiser='gd', optimiser_kwargs=optimiser_kwargs)
		mlp = MLP(shape=[(130107, 1500), 1500, (1500, 20), 20], dropout_proba=None, activation_fn=afn, max_epochs=100, validation_frequency=10, optimiser='gd', optimiser_kwargs=optimiser_kwargs)

		#mlp.W_flat_ = np.empty(4358002)


		#initialize.randomize_normal(mlp.W_flat_, 0, 1)
		#views = shaped_from_flat(mlp.W_flat_, mlp.shape_)
		#mlp.W_flat_ = ifn(views)

		#mlp.fit(X_train.toarray(), y_train, X_test.toarray(), y_test)
		mlp.fit(X_train, y_train, X_valid, y_valid)
		y_pred = mlp.predict(X_test)
		result_dict['{}_accuracy'.format(afn)] = accuracy_score(y_test, y_pred)
		result_dict['{}_f1_score'.format(afn)] = f1_score(y_test, y_pred, average='weighted' if len(np.unique(y_test)) > 2 else 'binary')

		# Plot learning curve & weight matrix
		#utils.plot_learning_curve(mlp.loss_history_, os.path.join(paths.get_out_path(), '20newsgroups', timestamped_foldername, 'results', 'learning_curve', afn))
		#utils.plot_weight_matrix(shaped_from_flat(mlp.W_best_flat_, mlp.shape_), os.path.join(paths.get_out_path(), '20newsgroups', timestamped_foldername, 'results', 'weight_matrices', afn))

	####################

	out_path = os.path.join(paths.get_out_path(), '20newsgroups', timestamped_foldername, 'results')
	fname = 'results.json'

	if (not os.path.exists(out_path)):
		os.makedirs(out_path)

	json.dump(result_dict, open(os.path.join(out_path, fname), 'w'))