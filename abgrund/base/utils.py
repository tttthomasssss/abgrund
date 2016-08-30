__author__ = 'thomas'
import os

from matplotlib import pyplot as plt
from matplotlib import pylab as pl
from sklearn.utils import shuffle

import numpy as np
import seaborn as sns


def one_hot(x, s=20):# 10
	if (x.ndim == 1):
		O = np.zeros((x.shape[0], s))
		O[list(range(x.shape[0])), x] = 1.

		return O
	else:
		return x


def num_instances(X):
	return len(X) if isinstance(X, list) else X.shape[0]


def gradient_check(nn, X, y, epsilon=0.001, error_threshold=0.01):
	#http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
	gradients = nn._backprop()


def plot_learning_curve(loss_history, plot_path):

	if (not os.path.exists(plot_path)):
		os.makedirs(plot_path)

	sns.set_style('whitegrid')

	plt.figure()
	plt.xlabel('Epochs')
	plt.ylabel('Validation Loss')
	plt.title('Validation Loss over time')
	plot_name = 'validation_loss_over_time.png'
	plt.grid(True)
	plt.plot(np.arange(1, len(loss_history) + 1), loss_history)
	plt.ylim([0., (max(loss_history) * 1.1)])
	plt.savefig(os.path.join(plot_path, plot_name))
	plt.close()


def plot_weight_matrix(views, plot_path):

	if (not os.path.exists(plot_path)):
		os.makedirs(plot_path)

	layer = 1
	for idx, W in enumerate(views, 1):
		if (W.ndim > 1):
			layer += 1

			# Colour
			plot_name = 'weight_matrix_at_layer_%d_colour.png' % (layer,)
			plt.figure()
			pl.matshow(W)
			plt.savefig(os.path.join(plot_path, plot_name))
			plt.close()

			# Black & White
			plot_name = 'weight_matrix_at_layer_%d_bw.png' % (layer,)
			plt.figure()
			pl.matshow(W, cmap=pl.cm.gray)
			plt.savefig(os.path.join(plot_path, plot_name))
			plt.close()


def create_random_state(random_state):
	if (isinstance(random_state, np.random.RandomState)):
		return random_state
	elif (isinstance(random_state, int)):
		return np.random.RandomState(seed=random_state)
	else:
		return np.random.RandomState(seed=1105)


def shuffle_mini_batch(X, Y):
	return shuffle(X, Y)


def dont_shuffle_mini_batch(X, Y):
	return (X, Y)