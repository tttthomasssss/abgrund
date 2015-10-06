__author__ = 'thomas'
import numpy as np


class VectorSpaceModel(object):
	'''
	Default VSM model to e.g. wrap gensim's word2vec and others
	'''
	def __init__(self, vsm, vector_shape, oov_handling='gaussian', random_state=np.random.RandomState(seed=1105), **kwargs):
		self.vsm_ = vsm
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
			'one': np.one
		}

	def __getitem__(self, item):
		if (item in self.vsm_ and self.oov_handling_ is not None):
			self.vsm_[item] = self.oov_handler_.get(self.oov_handling_, self.default_oov_handler_)(self.vector_shape_)

		return self.vsm_[item]

	def __contains__(self, item):
		return item in self.vsm_