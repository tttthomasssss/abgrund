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

from base import activation
from base import regularisation as reg
from base import utils
from vector_space.random_vectors import RandomVectorSpaceModel


# TODO:
#	- Support Multiple Layers
#	- Really nothing more than just started...
class AE(BaseEstimator):
	def __init__(self, shape, activation_fn='tanh', prediction_fn='softmax', W_init='xavier', gradient_check=True,
				 regularisation='l2', lambda_=0.01, dropout_proba=None, random_state=np.random.RandomState(seed=1105),
				 max_epochs=300, improvement_threshold=0.995, patience=np.inf, validation_frequency=100,
				 max_weight_norm=None, mini_batch_size=50, word_vector_dim=300, word_vector_model=None,
				 optimiser='gd', **optimiser_kwargs):
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

	def _lookup_word(self, w):
		return self.word_vector_model_[w].reshape(-1, 1)# if w in self.word_vector_model_ else np.ones((self.word_vector_dim_, 1))

	def _forward_propagation(self, x, W_enc, b_W_enc, W_dec, b_W_dec):
		curr_activations = []
		curr_predictions = []

		# Forward Propagation through whole sequence
		for x_i in x:

			# Lookup word vector in model
			v = self._lookup_word(x_i)

			### Encode ###

			# Linear Transformation
			a_enc = np.dot(W_enc.T, v) + b_W_enc

			# Apply Non-Linearity
			h_enc = self.activation_fn(a_enc)

			### Decode ###

			# Linear Transformation
			a_dec = np.dot(W_dec.T, h_enc) + b_W_dec

			# Apply Non-Linearity
			h_dec = self.activation_fn(a_dec)

			#h_dec is the prediction
			# Probably read this Tutorial first...http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/


			#curr_activations.insert(0, (v, a, h))
			#curr_predictions.insert(0, (p_a.T, p))

		return curr_activations, curr_predictions

	# Forward Propagation phase
	def predict_proba(self, X, W=None):
		W = self.W_flat_ if W is None else W
		views = shaped_from_flat(W, self.shape_)

		# Load weights for current epoch
		W_enc = views[0]
		b_W_enc = views[1].reshape(-1, 1)
		W_dec = views[2]
		b_W_dec = views[3].reshape(-1, 1)

		y_pred = []

		# For all Documents
		for x in X:
			_, pred = self._forward_propagation(x, W_enc, b_W_enc, W_dec, b_W_dec)
			y_pred.append(np.squeeze(pred[1][-1]))

		return np.array(y_pred)

	def predict(self, X):
		return np.argmax(self.predict_proba(X), axis=1)

	def loss(self, W, X, y):
		W = self.W_flat_ if W is None else W
		predictions = self.predict_proba(X, W)
		loss = (np.nan_to_num(-np.log(predictions)) * utils.one_hot(y, s=self.num_labels_)).sum(axis=1).mean() # use np.nansum for this utils.one_hot(y)).sum(axis=1).mean()

		# Add regularisation
		reg = self.regularisation_(self.lambda_, W, self.shape_)
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

	def _gradient_check(self, W, X, y, eps=10e-4): # TODO: Not quite right yet
		loss1 = self.loss(W + eps, X, y)
		loss2 = self.loss(W - eps, X, y)

		dLoss_dW_num = (loss1 - loss2) / 2 * eps

		return dLoss_dW_num

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
				return initialize.randomize_normal(self.W_flat_, random_state=self.random_state_)
		else: # Normal in [0 1]
			return initialize.randomize_normal(self.W_flat_, random_state=self.random_state_)

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

		# Load weights for current epoch
		W = views[0]
		b_W = views[1].reshape(-1, 1)
		V = views[2]
		b_V = views[3].reshape(-1, 1)

		dg_dW = np.zeros(W.shape)
		db_dW = np.zeros(b_W.shape)
		dg_dV = np.zeros(V.shape)
		db_dV = np.zeros(b_V.shape)

		# TODO: Re-Debug standard backprop in an MLP, then redo this with BPTT

		# Calculate Regularisation Gradients
		n_instances = utils.num_instances(X)
		dg_dV += self.deriv_regularisation_(self.lambda_, V) / n_instances
		dg_dW += self.deriv_regularisation_(self.lambda_, W) / n_instances

		# Prediction of network w.r.t. to current W
		for i, x in enumerate(X):
			activations, predictions = self._forward_propagation(x, W, b_W, V, b_V)

			# Error in last layer for current instance
			y_pred_activation, y_pred = predictions[0][0], predictions[0][1] # TODO: Maybe use output of every node?
			e = self.deriv_prediction_fn(y_pred, y[i]).reshape(1, -1)
			delta_L = e * self.deriv_activation_fn(y_pred_activation)

			a_last = activations[0][-1]

			# Gradient from BPTT
			dg_dV += np.dot(a_last, delta_L)
			db_dV += delta_L.T

			delta_one_lower_x = delta_L
			delta_one_lower_a = delta_L
			W_one_lower = V
			for j in range(1, len(activations) - 1):
				_, _, h_out = activations[j]
				x_in, a_in, _ = activations[j + 1]

				# The messyness sets in when we change from V to W as the backprop weight
				# With the backprop weight being W, we need to handle the recurrent input as well as the word vector input
				# and hence, the 2-weights-in-1 backfires during backprop (badumm-ts) and makes things a little ugly
				if (j > 1):
					delta_lx = np.dot(W_one_lower[:self.word_vector_dim_, :], delta_one_lower_x) * self.deriv_activation_fn(h_out)
					delta_la = np.dot(W_one_lower[self.word_vector_dim_:, :], delta_one_lower_a) * self.deriv_activation_fn(h_out)
				else:
					delta_lx = np.dot(W_one_lower, delta_one_lower_x.T) * self.deriv_activation_fn(h_out)
					delta_la = delta_lx

				# 2-part gradient update, 1) from word vector input, 2) from recurrent input
				dg_dW[:self.word_vector_dim_, :] += np.dot(x_in, delta_lx.T)
				dg_dW[self.word_vector_dim_:, :] += np.dot(a_in, delta_la.T)

				db_dW += delta_lx # Only the input has a bias, the recurrent layer has not!

				delta_one_lower_x = delta_lx
				delta_one_lower_a = delta_la
				W_one_lower = W

		return np.concatenate([dg_dW.flatten(), db_dW.flatten(), dg_dV.flatten(), db_dV.flatten()])

if (__name__ == '__main__'):
	### Bag of Words
	data = dataset_utils.fetch_stanford_sentiment_treebank_dataset()

	y_train, y_valid, y_test = data[1], data[3], data[5]

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

	### Word Vectors
	#w2c = dataset_utils.fetch_google_news_word2vec_300dim_vectors()
	vsm = RandomVectorSpaceModel()
	#vsm.construct(data[0])
	gd_params = {'step_rate': 1., 'momentum': 0.95, 'momentum_type': 'nesterov'}
	ae = AE(shape=[(50, 20), 20, (20, 50), 50], activation_fn='tanh', max_epochs=200, validation_frequency=10,
			  word_vector_dim=300, word_vector_model=vsm, mini_batch_size=-1, optimiser='gd', **gd_params)

	#rnn.predict_proba(data[0][:1])
	ae.fit(data[0], y_train, data[2], y_valid)
	y_pred = ae.predict_proba(data[0])

	print('[RNN VSM] Accuracy: %f; F1-Score: %f' % (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')))
