from __future__ import division
__author__ = 'thomas'
import numpy as np
'''
Some Softmax resources:
https://gist.github.com/stober/1946926
http://www.johndcook.com/blog/2010/01/20/how-to-compute-the-soft-maximum/
http://en.wikipedia.org/wiki/Softmax_function
https://chrisjmccormick.wordpress.com/2014/06/13/deep-learning-tutorial-softmax-regression/
'''


def softmax(X, W):
	f = np.dot(W, X)
	f -= np.max(f) # Numerical stability

	return np.exp(f) / np.sum(np.exp(f))


if (__name__ == '__main__'):
	X = np.matrix([[1, 2, 3], [0.1, 0.01, 0.001], [3, 7, 2]])
	W = np.matrix([[4, 1, 6], [7, 8, 9], [4, -4, 1]])

	print 'SOFTMAX:', softmax(X, W)
	print 'SOFTMAX SUM:', np.sum(softmax(X, W))