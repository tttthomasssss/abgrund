__author__ = 'thomas'
import itertools

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
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
class CNN(BaseEstimator):
	def __init__(self, shape, vector_space_model, activation_fn='tanh', prediction_fn='softmax', w_init='xavier',
				 gradient_check=False, regularisation='l2', lambda_=0.01, dropout_proba=None,
				 random_state=np.random.RandomState(seed=1105), max_epochs=20, improvement_threshold=0.995,
				 patience=np.inf, validation_frequency=100, max_weight_norm=None, mini_batch_size=50, optimiser='gd',
				 optimiser_kwargs={}, filter_ngram_ranges=[2, 3], num_filters=100):
		self.random_state_ = utils.create_random_state(random_state)
		self.shape_ = shape
		self.vector_space_model_ = vector_space_model
		self.filter_ngram_ranges_ = filter_ngram_ranges
		self.num_unique_filters_ = len(filter_ngram_ranges)
		self.num_filters_ = num_filters
		self.feature_maps_ = []
		self.max_pooling_ = []
		self.conv_weights_ = self._initialise_conv_weights(w_init, activation_fn)
		self.weights_ = self._initialise_weights(w_init, activation_fn, self.shape_)
		self.activation_fn, self.deriv_activation_fn = (getattr(activation, activation_fn), getattr(activation, 'deriv_{}'.format(activation_fn)))
		self.prediction_fn, self.deriv_prediction_fn = (getattr(activation, prediction_fn), getattr(activation, 'deriv_{}'.format(prediction_fn)))
		optimiser_kwargs['shape'] = self.shape_
		self.optimiser_ = getattr(optimisation, optimiser)(**optimiser_kwargs)
		self.gradient_check_ = gradient_check
		self.lambda_ = lambda_
		self.regularisation_, self.deriv_regularisation_ = (getattr(reg, '{}_regularisation'.format(regularisation)), getattr(reg, 'deriv_{}_regularisation'.format(regularisation)))
		self.max_weight_norm_ = max_weight_norm
		self.dropout_proba_ = self._create_dropout_prediction_proba(dropout_proba)
		self.dropout_masks_ = self._create_dropout_masks(dropout_proba) # Be careful with dropout_proba, this fn modifies it
		self.max_epochs_ = np.inf if max_epochs is None or max_epochs < 0 else max_epochs
		self.improvement_threshold_ = improvement_threshold
		self.patience_ = patience
		self.validation_frequency_ = validation_frequency
		self.loss_history_ = []
		self.mini_batch_size_ = mini_batch_size
		self.num_classes_ = 0
		self.num_instances_ = 0

	# Forward Propagation phase
	def _forward_propagation(self, doc, dropout_mode='fit'):
		X = self.vector_space_model_.concatenate_vectors(doc)

		for idx, filter in enumerate(self.filter_ngram_ranges_):

			# Convolution
			for w, b in self.conv_weights_[idx]:
				feature_map = []
				for i in range(X.shape[0] - filter + 1):
					feature_map.append(self.activation_fn(np.sum(w * X[i:i+filter]) + b)) # X[i:i+filter-1]), python indexes [a, b)

				# Max Pooling
				self.max_pooling_.append(np.max(feature_map))

		'''
		self.max_pooling_ contains all of the maxed features that represent the penultimate feature vector to be passed into the softmax classifier
		'''

		# Concatenate Pooling Features and predict
		idx = list(range(len(self.weights_)))
		b_pred = self.weights_[idx.pop()]
		W_pred = self.weights_[idx.pop()]

		while len(idx) > 1:
			pass



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

	def fit(self, docs, y, docs_valid, y_valid):
		# Some initial administrative stuff
		self.num_classes_ = np.unique(y).shape[0]
		self.num_instances_ = utils.num_instances(docs)
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

				self.weights_ = self.optimiser_(weights=self.weights_, gradients=gradients)

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

	def _initialise_weights(self, w_init, activation_fn, shape):
		if (w_init == 'xavier' and hasattr(initialisation, 'randomise_uniform_{}'.format(activation_fn))):
			return getattr(initialisation, 'randomise_uniform_{}'.format(activation_fn))(shape)
		else:
			return initialisation.randn(self.shape_)

	def _initialise_conv_weights(self, w_init, activation_fn):
		conv_weights = []
		for idx, ngram_range in enumerate(self.filter_ngram_ranges_):
			conv_weights.append([])
			for i in range(self.num_filters_):
				w = self._initialise_weights(w_init, activation_fn, [(ngram_range, self.vector_space_model_.vector_shape)])[0]
				conv_weights[idx].append((w, 1)) # TODO: remove the hardcoded bias term...

		return conv_weights

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