__author__ = 'thomas'
from collections import OrderedDict

import numpy as np


class RandomVectorSpaceModel(object):

	def __init__(self, ndim=50, random_state=np.random.RandomState(seed=1105)):
		self.ndim_ = ndim
		self.random_state_ = random_state
		self.vsm_ = {}
		self.vocab_ = set()

	def construct(self, data, initialise_immediately=False):
		self.vocab_ = reduce(lambda vocab, doc: vocab | set(doc), data, set())

		if (initialise_immediately):
			for w in self.vocab_:
				self.vsm_[w] = self.random_state_.randn(self.ndim_,)
		print len(self.vocab_)

	def __getitem__(self, item):
		if (item not in self.vsm_):
			self.vsm_[item] = self.random_state_.randn(self.ndim_,)

		return self.vsm_[item]

	def __contains__(self, item):
		return item in self.vsm_

	def asarray(self, dtype=np.float64):
		ordered_vsm = OrderedDict(sorted(self.vsm_.items()))
		return ordered_vsm.keys(), np.array(ordered_vsm.values(), dtype=dtype)