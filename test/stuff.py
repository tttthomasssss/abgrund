__author__ = 'thk22'
import climin
import numpy as np

from base import utils


def test_minibatches():
	N = 100000 #100 # Number of Samples per Class
	D = 2 # Number of Features (dimensionality of data)
	K = 3 # Number of Classes
	X = np.zeros((N * K, D))
	y = np.zeros(N * K, dtype='uint8')
	num_examples = X.shape[0]
	for j in range(K):
		ix = np.arange(N * j, N * (j + 1))

		r = np.linspace(0.0, 1, N)
		t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
		X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
		y[ix] = j

	for idx, (args, kwargs) in enumerate(((i, {}) for i in climin.util.iter_minibatches([X, utils.one_hot(y, s=3)], 200, [0, 0])), 1):
		if (len(args[0]) <= 0):
			print('EMPTY AT: {}'.format(idx))
		elif (idx % 10000 == 0):
			print('STILL ALL GOOD AT: {}'.format(idx))

if (__name__ == '__main__'):
	test_minibatches()