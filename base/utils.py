__author__ = 'thomas'
import os

from climin import Adadelta
from climin import Adam
from climin import ConjugateGradient
from climin import GradientDescent
from climin import Lbfgs
from climin import NonlinearConjugateGradient
from climin import RmsProp
from climin import Rprop
from matplotlib import pyplot as plt
from matplotlib import pylab as pl
import numpy as np
import seaborn as sns


def one_hot(x, s=20):# 10
	if (x.ndim == 1):
		O = np.zeros((x.shape[0], s))
		O[xrange(x.shape[0]), x] = 1.

		return O
	else:
		return x


def num_instances(X):
	if (isinstance(X, list)):
		return len(X)
	else:
		return X.shape[0]


def get_optimiser_for_string(optimiser,  **optimiser_kwargs):
	if (optimiser == 'gd'):
		return GradientDescent(**optimiser_kwargs)
	if (optimiser == 'lbfgs'):
		return Lbfgs(**optimiser_kwargs)
	if (optimiser == 'rmsprop'):
		return RmsProp(**optimiser_kwargs)
	if (optimiser == 'adadelta'):
		return Adadelta(**optimiser_kwargs)
	if (optimiser == 'adam'):
		return Adam(**optimiser_kwargs)
	if (optimiser == 'cg'):
		return ConjugateGradient(**optimiser_kwargs)
	if (optimiser == 'nonlinearcg'):
		return NonlinearConjugateGradient(**optimiser_kwargs)
	if (optimiser == 'rprop'):
		return Rprop(**optimiser_kwargs)


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