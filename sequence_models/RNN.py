__author__ = 'thomas'
import pickle
import collections
import itertools
import json
import os

from climin import GradientDescent
from climin import NonlinearConjugateGradient
from climin import Lbfgs
from climin import Rprop
from climin import RmsProp
from climin import initialize
from climin.util import empty_with_views
from climin.util import iter_minibatches
from climin.util import shaped_from_flat
from common import dataset_utils
from common import paths
from preprocessing.data_preparation import split_data
from preprocessing.data_preparation import split_data_train_dev_test
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import safe_sparse_dot
from utils import path_utils
import numpy as np
import scipy as sp

from base import activation
from base import regularisation as reg
from base import utils
from vector_space.random_vectors import RandomVectorSpaceModel


# TODO:
#	- Dropout for RNN
#	- Backprop into vectors (https://github.com/vsl9/Sentiment-Analysis-with-Convolutional-Networks/issues/2 or check keras source)
#	- stick softmax on every unit to have local gradients
class RNN(BaseEstimator):
	def __init__(self, shape, activation_fn='tanh', prediction_fn='softmax', W_init='xavier', gradient_check=True,
				 regularisation='l2', lambda_=0.01, dropout_proba=None, random_state=np.random.RandomState(seed=1105),
				 max_epochs=300, improvement_threshold=0.995, patience=np.inf, validation_frequency=100,
				 max_weight_norm=None, mini_batch_size=50, word_vector_dim=300, word_vector_model=None,
				 update_word_vectors=False, optimiser='gd', **optimiser_kwargs):
		self.random_state_ = self._create_random_state(random_state)
		self.activations_ = []
		self.predictions_ = []
		self.shape_ = shape
		self.W_flat_ = self._initialise_weights(W_init, activation_fn)
		self.activation_fn, self.deriv_activation_fn = activation.get_activation_fn_for_string(activation_fn)
		self.prediction_fn, self.deriv_prediction_fn = activation.get_prediction_fn_for_string(prediction_fn)
		self.optimiser = optimiser
		self.optimiser_kwargs = optimiser_kwargs
		self.gradient_check_ = gradient_check
		self.lambda_ = lambda_
		self.regularisation_, self.deriv_regularisation_ = reg.get_regularisation_fn_for_string(regularisation)
		self.max_weight_norm_ = max_weight_norm
		self.dropout_masks_ = self._create_dropout_masks(dropout_proba)
		self.max_epochs_ = np.inf if max_epochs is None or max_epochs < 0 else max_epochs
		self.improvement_threshold_ = improvement_threshold
		self.patience_ = patience
		self.W_best_flat_ = np.array([])
		self.validation_frequency_ = validation_frequency
		self.loss_history_ = []
		self.mini_batch_size_ = mini_batch_size
		self.num_labels_ = 0
		self.h_initial_ = np.zeros((word_vector_dim, 1), dtype=np.float64)
		self.word_vector_model_ = word_vector_model
		self.word_vector_dim_ = word_vector_dim
		self.update_word_vectors = update_word_vectors

	# Forward Propagation phase
	def _forward_propagation(self, doc, W, b_W, V, b_V):
		activations = []
		magnitudes = []

		# Initial hidden state
		h = self.h_initial_

		# Forward Propagation through whole sequence
		for w_i in doc:

			# Lookup word vector in model
			v = self.word_vector_model_[w_i].reshape(-1, 1)

			# Add word vectors to activations (only used if self.update_word_vectors == True)
			activations.append(v)

			# Stack current input word vector (x_i) and previous hidden state h (t_i-1) together
			s = np.concatenate((v, h))

			# Linear Transformation (W contains the weights of the recurrent input as well as the standard input)
			z = safe_sparse_dot(W.T, s) + b_W
			magnitudes.append(z)

			# Apply Non-Linearity
			h = self.activation_fn(z)
			activations.append(h)

		# Prediction - Linear Transformation
		z = safe_sparse_dot(V.T, activations[-1]) + b_V

		# Predict!
		a = self.prediction_fn(z.T)

		activations.append(a)
		magnitudes.append(z)

		return activations, magnitudes

	def predict_proba(self, X, W=None):
		W = self.W_flat_ if W is None else W
		views = shaped_from_flat(W, self.shape_)

		# Load weights for current epoch (views[0] are the lookup weights)
		W = views[1]
		b_W = views[2].reshape(-1, 1)
		V = views[3]
		b_V = views[4].reshape(-1, 1)

		y_pred = []

		# For all Documents
		for doc in X:
			activations, _ = self._forward_propagation(doc, W, b_W, V, b_V)
			y_pred.append(np.squeeze(activations[-1]))

		return np.array(y_pred)

	def predict(self, X):
		return np.argmax(self.predict_proba(X), axis=1)

	def loss(self, W, X, y):
		W = self.W_flat_ if W is None else W
		predictions = self.predict_proba(X, W)
		loss = (np.nan_to_num(-np.log(predictions)) * utils.one_hot(y, s=self.num_labels_)).sum(axis=1).mean() # use np.nansum for this utils.one_hot(y)).sum(axis=1).mean()

		# Add regularisation
		reg = self.regularisation_(self.lambda_, W, self.shape_, skip_first=True)
		loss += (reg / (utils.num_instances(X) * 2))

		return loss

	def fit(self, X, y, X_valid=None, y_valid=None):
		self.num_labels_ = np.unique(y).shape[0]
		args = self._get_input_chain(X, y)
		self.optimiser_kwargs['args'] = args
		self.optimiser_kwargs['wrt'] = self.W_flat_
		self.optimiser_kwargs['fprime'] = self.dLoss_dW
		if (self.optimiser in ['lbfgs', 'nonlinearcg']):
			self.optimiser_kwargs['f'] = self.loss

		opt = utils.get_optimiser_for_string(self.optimiser, **self.optimiser_kwargs)

		#W_history = []
		self.loss_history_ = []
		min_loss = np.inf
		min_loss_idx = np.inf
		curr_patience = self.patience_
		validation_loss = np.inf
		self.W_best_flat_ = self.W_flat_

		print('### %s ###' % (self.optimiser,))
		print('Init Loss:', self.loss(None, X_valid, y_valid))
		y_pred = np.argmax(self.predict_proba(X_valid), axis=1)
		print('Init Accuracy:', accuracy_score(y_valid, y_pred))
		for info in opt:
			# Check for validation loss
			if (info['n_iter'] % self.validation_frequency_ == 0):
				validation_loss = self.loss(opt.wrt, X_valid, y_valid)
				self.loss_history_.append(validation_loss)
				print('\tIteration=%d; Training Loss=%.4f; Validation Loss=%.4f[patience=%r]' % (info['n_iter'], self.loss(None, X, y), validation_loss, curr_patience))
				#W_history.append(opt.wrt)

				curr_patience -= 1

				if (validation_loss < min_loss * self.improvement_threshold_):
					min_loss = validation_loss
					#min_loss_idx = len(W_history) - 1
					self.W_best_flat_ = opt.wrt
					curr_patience = self.patience_

			# Max Norm constraint, see Hinton (2012) or Kim (2014) - often used in conjunction with dropout
			if (self.max_weight_norm_ is not None):
				curr_norm = np.linalg.norm(self.W_flat_)
				if (curr_norm > self.max_weight_norm_):
					self.W_flat_ *= (self.max_weight_norm_ / np.linalg.norm(self.W_flat_))

			# Stopping criterion met?
			if (self._stopping_criterion(info['n_iter'], curr_patience, validation_loss)):
				break

		y_pred = np.argmax(self.predict_proba(X_valid, W=opt.wrt), axis=1)
		print('Final Loss:', self.loss(opt.wrt, X_valid, y_valid))
		print('Final Accuracy:', accuracy_score(y_valid, y_pred))

		#y_pred = np.argmax(mlp.predict_proba(X_valid, W_history[min_loss_idx]), axis=1)
		#y_pred_train = np.argmax(mlp.predict_proba(X, W_history[min_loss_idx]), axis=1)
		#print 'Optimal Loss:', mlp.loss(W_history[min_loss_idx], X_valid, y_valid)
		#print 'Final Accuracy:', accuracy_score(y_valid, y_pred)
		#print 'Final Accuracy Train:', accuracy_score(y, y_pred_train)
		y_pred = np.argmax(self.predict_proba(X_valid, self.W_best_flat_), axis=1)
		y_pred_train = np.argmax(self.predict_proba(X, self.W_best_flat_), axis=1)
		print('Optimal Loss:', self.loss(self.W_best_flat_, X_valid, y_valid))
		print('Final Accuracy:', accuracy_score(y_valid, y_pred))
		print('Final Accuracy Train:', accuracy_score(y, y_pred_train))
		self.W_flat_ = self.W_best_flat_

	def _stopping_criterion(self, curr_iter, curr_patience, loss):
		if (np.isinf(curr_patience)):
			return curr_iter > self.max_epochs_ or loss <= 0
		else:
			return curr_patience <= 0 or loss <= 0

	def _get_input_chain(self, X, y):
		if (self.mini_batch_size_ <= 0):
			return itertools.repeat(([X, utils.one_hot(y, s=self.num_labels_)], {}))
		else:
			return ((i, {}) for i in iter_minibatches([X, utils.one_hot(y, s=self.num_labels_)], self.mini_batch_size_, [0, 0]))

	def _gradient_check(self, W, X, y, eps=10e-4, error_threshold=10e-2):
		dg_dW = self.dLoss_dW(W, X, utils.one_hot(y, self.num_classes_))

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
		self.W_flat_ = np.empty(self._count_params())
		views = shaped_from_flat(self.W_flat_, self.shape_)
		if (W_init == 'xavier'):
			if (activation_fn == 'sigmoid'):
				return initialize.randomize_uniform_sigmoid(views, self.random_state_)
			elif (activation_fn == 'tanh'):
				return initialize.randomize_uniform_tanh(views, self.random_state_)
			elif (activation_fn == 'relu'):
				return initialize.randomize_uniform_relu(views, self.random_state_)
			else:
				return initialize.randn(views, random_state=self.random_state_, scale=0.01)

			# Add Word Vectors to first layer
			#vsm.asarray()
		else: # Normal in [0 1]
			return initialize.randn(views, random_state=self.random_state_, scale=0.01)

	def _count_params(self):
		n_params = 0
		for x in self.shape_:
			if (isinstance(x, collections.Iterable)):
				n_params += (x[0] * x[1])
			else:
				n_params += x

		return n_params

	def _create_random_state(self, random_state):
		if (isinstance(random_state, np.random.RandomState)):
			return random_state
		elif (isinstance(random_state, int)):
			return np.random.RandomState(seed=random_state)
		else:
			return np.random.RandomState(seed=1105)

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

	def dLoss_dW(self, W, X, y):
		views = shaped_from_flat(W, self.shape_)
		self.W_flat_ = W

		# Load weights for current epoch (views[0] contains the lookup layer)
		W = views[1]
		b_W = views[2].reshape(-1, 1)
		V = views[3]
		b_V = views[4].reshape(-1, 1)

		dg_dX = np.zeros((len(self.word_vector_model_), self.word_vector_dim_)) # Init gradients for word vectors
		dg_dW = np.zeros(W.shape)
		db_dW = np.zeros(b_W.shape)
		dg_dV = np.zeros(V.shape)
		db_dV = np.zeros(b_V.shape)

		# Calculate Regularisation Gradients
		n_instances = utils.num_instances(X)
		dg_dV += self.deriv_regularisation_(self.lambda_, V) / n_instances
		dg_dW += self.deriv_regularisation_(self.lambda_, W) / n_instances

		# Prediction of network w.r.t. to current W
		for i, doc in enumerate(X):
			activations, magnitudes = self._forward_propagation(doc, W, b_W, V, b_V)

			# Error in last layer for current instance
			y_pred = activations.pop()
			z = magnitudes.pop()

			# BP 1: Error in last layer
			delta_l = self.deriv_prediction_fn(y_pred, y[i].reshape(1, -1)).T * self.deriv_activation_fn(z)

			# Pop another activation
			a = activations.pop()

			# Gradients w.r.t. last layer error
			dg_dV += safe_sparse_dot(a, delta_l.T) / n_instances # BP 4: dot product between inputs that caused the error and backpropped error
			db_dV += delta_l / n_instances # BP 3: gradient of bias = delta_l

			# Handle penultimate layer separately as well, this is because things get messy when we change from V to W
			# as the backprop weight!
			# With the backprop weight being W, we need to handle the recurrent input as well as the word vector input
			# and hence, the 2-weights-in-1 backfires during backprop (badumm-ts) and makes things a little ugly
			delta_one_lower_x = delta_l
			W_one_lower = V

			z = magnitudes.pop()
			a_word_vector = activations.pop()
			a_in = activations.pop()

			delta_lx = safe_sparse_dot(W_one_lower, delta_one_lower_x) * self.deriv_activation_fn(z)
			delta_la = delta_lx

			# 2-part gradient update, 1) from word vector input, 2) from recurrent input
			dg_dW[:self.word_vector_dim_, :] += safe_sparse_dot(a_word_vector, delta_lx.T) # TODO: Try sequence length normalised update
			dg_dW[self.word_vector_dim_:, :] += safe_sparse_dot(a_in, delta_la.T)

			db_dW += delta_lx # Only the input has a bias, the recurrent layer has not!

			# Backpropagate error signal from x_in to word vector as well
			# Should just be delta_lx! --> dot product with indicator matrix (vocab_size x 1), to get the gradient for the full word vector table (vocab_size x vector_dim)

			delta_one_lower_x = delta_lx
			delta_one_lower_a = delta_la
			W_one_lower = W

			# Finally, BPTT through the remaining timesteps
			while (len(activations) > 1):
				# Pop the activations and magnitude
				a_word_vector = activations.pop()
				a_in = activations.pop()
				z = magnitudes.pop()

				delta_lx = np.dot(W_one_lower[:self.word_vector_dim_, :], delta_one_lower_x) * self.deriv_activation_fn(z)
				delta_la = np.dot(W_one_lower[self.word_vector_dim_:, :], delta_one_lower_a) * self.deriv_activation_fn(z)

				# 2-part gradient update, 1) from word vector input, 2) from recurrent input
				dg_dW[:self.word_vector_dim_, :] += safe_sparse_dot(a_word_vector, delta_lx.T)
				dg_dW[self.word_vector_dim_:, :] += safe_sparse_dot(a_in, delta_la.T)

				db_dW += delta_lx # Only the input has a bias, the recurrent layer has not!

				# Backpropagate error signal from x_in to word vector as well
				# Should just be delta_lx! --> dot product with indicator matrix (vocab_size x 1), to get the gradient for the full word vector table (vocab_size x vector_dim)

				delta_one_lower_x = delta_lx
				delta_one_lower_a = delta_la

			# Backpropagate error signal from x_in to word vector as well
			# Should just be delta_lx! --> dot product with indicator matrix (vocab_size x 1), to get the gradient for the full word vector table (vocab_size x vector_dim)
			# pop last word vector & update

		return np.concatenate([dg_dX.flatten(), dg_dW.flatten(), db_dW.flatten(), dg_dV.flatten(), db_dV.flatten()])

if (__name__ == '__main__'):
	### Bag of Words
	data = dataset_utils.fetch_stanford_sentiment_treebank_dataset()

	y_train, y_valid, y_test = data[1], data[3], data[5]

	'''
	vec = CountVectorizer()
	X_train = vec.fit_transform([' '.join(l) for l in data[0]])
	X_valid = vec.transform([' '.join(l) for l in data[2]])
	X_test = vec.transform([' '.join(l) for l in data[4]])

	mnb = MultinomialNB()
	mnb.fit(X_train, y_train)
	y_pred = mnb.predict(X_test)

	print('[MNB BoW] Accuracy: %f; F1-Score: %f' % (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')))

	svm = LinearSVC()
	svm.fit(X_train, y_train)
	y_pred = svm.predict(X_test)

	print('[SVM BoW] Accuracy: %f; F1-Score: %f' % (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')))
	'''
	### Word Vectors
	#w2c = dataset_utils.fetch_google_news_word2vec_300dim_vectors()
	vsm = RandomVectorSpaceModel()
	vsm.construct(data[0])
	gd_params = {'step_rate': 1., 'momentum': 0.95, 'momentum_type': 'nesterov'}
	#rnn = RNN(shape=[(100, 50), 50, (50, 5), 5], activation_fn='tanh', max_epochs=200, validation_frequency=10,
	#		  word_vector_dim=50, word_vector_model=vsm, mini_batch_size=-1, optimiser='gd', **gd_params)

	rnn = RNN(shape=[(len(vsm), 50), (100, 50), 50, (50, 5), 5], activation_fn='tanh', max_epochs=200, validation_frequency=10,
			  word_vector_dim=50, word_vector_model=vsm, mini_batch_size=-1, optimiser='gd', **gd_params)

	#rnn.predict_proba(data[0][:1])
	rnn.fit(data[0], y_train, data[2], y_valid)
	y_pred = rnn.predict(data[4])

	print('[RNN VSM] Accuracy: %f; F1-Score: %f' % (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')))
