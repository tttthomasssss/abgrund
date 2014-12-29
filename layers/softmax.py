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


def softmax(x, W, j):
	print np.dot(W[j, :], x)
	print 'ENUMERATOR:', np.exp(np.dot(W[j, :], x))
	print np.dot(W, x)
	print np.exp(np.dot(W, x))
	print 'DENOMINATOR:',  np.sum(np.exp(np.dot(W, x)))
	return np.exp(np.dot(W[j, :], x)) / np.sum(np.exp(np.dot(W, x)))


if (__name__ == '__main__'):
	X = np.matrix([[1, 2, 3], [0.1, 0.01, 0.001], [3, 3, 2]])
	W = np.matrix([[4, 5, 6], [7, 8, 9], [4, 2, 1]])

	lll = 0.
	for i in xrange(3):
		lll+=softmax(X[:, i], W, i)
	print 'SOFTMAX:',lll