__author__ = 'thomas'
import numpy as np


class VectorSpaceModel(object):
	'''
	Default VSM model to e.g. wrap gensim's word2vec and others
	'''
	def __init__(self, vsm, vector_shape, vsm_type, oov_handling='gaussian', random_state=np.random.RandomState(seed=1105), **kwargs):
		self.vsm_ = vsm
		self.vsm_type_ = vsm_type
		self.vector_shape_ = vector_shape
		self.random_state_ = random_state
		self.kwargs_ = kwargs
		self.default_oov_handler_ = self.random_state_.randn
		self.oov_handling_ = oov_handling
		self.oov_handler_ = {
			'gaussian': self.random_state_.randn,
			'bernoulli': self.random_state_.binomial,
			'uniform': self.random_state_.uniform,
			'zero': np.zeros,
			'one': np.ones
		}
		self.oov_ = {}

	@property
	def vector_shape(self):
		return self.vector_shape_

	def __getitem__(self, item):
		return getattr(self, '_{}_getitem'.format(self.vsm_type_))(item)

	def __contains__(self, item):
		return getattr(self, '_{}_contains'.format(self.vsm_type_))(item)

	def _random_contains(self, item):
		return item in self.vsm_

	def _random_getitem(self, item):
		if (item not in self.vsm_ and self.oov_handling_ is not None):
			if (item not in self.oov_):
				oov = self.oov_handler_.get(self.oov_handling_, self.default_oov_handler_)(self.vector_shape_)
				self.oov_[item] = oov
			else:
				oov = self.oov_[item]

			return oov
		else:
			return self.vsm_[item]

	def _word2vec_contains(self, item):
		return item in self.vsm_

	def _word2vec_getitem(self, item):
		if (item not in self.vsm_ and self.oov_handling_ is not None):
			if (item not in self.oov_):
				oov = self.oov_handler_.get(self.oov_handling_, self.default_oov_handler_)(self.vector_shape_)
				self.oov_[item] = oov
			else:
				oov = self.oov_[item]

			return oov
		else:
			return self.vsm_[item]

	def _glove_contains(self, item):
		return item in self.vsm_.dictionary

	def _glove_getitem(self, item):
		if (item not in self.vsm_.dictionary and self.oov_handling_ is not None):
			if (item not in self.oov_):
				oov = self.oov_handler_.get(self.oov_handling_, self.default_oov_handler_)(self.vector_shape_)
				self.oov_[item] = oov
			else:
				oov = self.oov_[item]

			return oov
		else:
			return self.vsm_.word_vectors[self.vsm_.dictionary[item]]

	def transform(self, documents, composition=np.sum, p_keep_word=1.0):
		X = np.zeros((len(documents), self.vector_shape_))

		for idx, document in enumerate(documents):
			A = np.zeros((len(document), self.vector_shape_))

			for jdx, w in enumerate(document):
				A[jdx] = self[w.strip()]

			# Word Dropout
			keep = self.random_state_.binomial(1, p_keep_word, (len(document),))
			A *= keep[:, np.newaxis]

			X[idx] = composition(A, axis=0)

		return X

	def concatenate_vectors(self, document, p_keep_word=1.0):

		# Sample initial document
		idx = 0
		if (p_keep_word < 1.0):
			sampled = False
			idx = -1

			while not sampled:
				sampled = (self.random_state_.binomial(1, p_keep_word, (1,))[0] == 1)
				idx += 1

		X = self.vsm_[document[idx]]

		# Concatenate rest with word dropout sampling
		for w in document[idx+1:]:
			if (self.random_state_.binomial(1, p_keep_word, (1,))[0] == 1):
				X = np.vstack((X, self.vsm_[w]))

		return X