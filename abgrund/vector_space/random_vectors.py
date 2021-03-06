__author__ = 'thomas'
from collections import OrderedDict

import numpy as np
from functools import reduce


class RandomVectorSpaceModel(object):

	def __init__(self, ndim=50, random_state=np.random.RandomState(seed=1105), random_sampler='randn'):
		self.ndim_ = ndim
		self.random_state_ = random_state
		self.random_sampler_ = random_sampler
		self.vsm_ = {}
		self.vocab_ = set()
		self.vocab_size_ = None
		self.inverted_index = {}

	def construct(self, data, initialise_immediately=False):
		self.vocab_ = sorted(reduce(lambda vocab, doc: vocab | set(doc), data, set()))

		if (initialise_immediately):
			for idx, w in enumerate(self.vocab_):
				self.vsm_[w] = getattr(self.random_state_, self.random_sampler_)(self.ndim_,)
				self.inverted_index[w] = idx

			self._vocab_size_ = len(self.vocab_)

	def __len__(self):
		return self.vocab_size_ if self.vocab_size_ is not None else len(self.vocab_)

	def __getitem__(self, item):
		if (item not in self.vsm_):
			self.vsm_[item] = self.random_state_.randn(self.ndim_,)

		return self.vsm_[item]

	def __contains__(self, item):
		return item in self.vsm_

	def dimensionality(self):
		return self.ndim_

	def index(self, w):
		if (len(self.inverted_index) <= 0):
			raise NotImplementedError
		else:
			return self.inverted_index[w]

	def asarray(self, dtype=np.float64):
		ordered_vsm = OrderedDict(sorted(self.vsm_.items()))
		return list(ordered_vsm.keys()), np.array(list(ordered_vsm.values()), dtype=dtype)