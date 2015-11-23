__author__ = 'thomas'
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import climin
import numpy as np

from base import utils
from mlp.feedforward import MLP


def test_mlp_toy_dataset():
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

	#### PLOT DATA
	plt.figure()
	plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
	plt.savefig('/Volumes/LocalDataHD/thk22/DevSandbox/InfiniteSandbox/tag-lab/abgrund/test/dump/data.png')
	plt.close()
	####

	# eta=1.0; layer_size=200; lambda=0.1; activation=relu; weight_initialisation=xavier (batch mode)###
	'''
	gd_params = {
			'step_rate': 1.0
	}

	mlp = MLP(shape=[(2, 200), 200, (200, 3), 3], dropout_proba=None, activation_fn='relu', max_epochs=100000, gradient_check=False,
								  validation_frequency=1000, lambda_=0.1, mini_batch_size=200, W_init='xavier', optimiser='gd', **gd_params)

	mlp.fit(X, y)

	views = mlp.best_weights()
	W1 = views[0]
	b1 = views[1]
	W2 = views[2]
	b2 = views[3]

	#### PLOT DECISION BOUNDARY
	# plot the resulting classifier
	h = 0.02
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						 np.arange(y_min, y_max, h))
	Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W1) + b1), W2) + b2
	Z = np.argmax(Z, axis=1)
	Z = Z.reshape(xx.shape)
	fig = plt.figure()
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
	plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	fig.savefig('/Volumes/LocalDataHD/thk22/DevSandbox/InfiniteSandbox/tag-lab/abgrund/test/dump/mlp_decision_boundary.png')
	####
	'''
	''' Hyperparam Tuning '''
	best_acc = -np.inf
	best_config = []
	for sr in [1., 0.1, 0.01, 0.001]:
		for hh in [30, 50, 100, 200, 500]:
			for l in [0.001, 0.01, 0.1, 1.]:
				for afn in ['relu', 'sigmoid', 'tanh']:
					for wi in ['xavier', 'randn']:
						for mbs in [10000, 500, 200, 50]:

							print('### CONFIG: eta={}; layer_size={}; lambda={}; activation={}; weight_initialisation={}; mini_batch_size={} ###'.format(sr, hh, l, afn, wi, mbs))

							gd_params = {
								'step_rate': sr
							}

							mlp = MLP(shape=[(2, hh), hh, (hh, 3), 3], dropout_proba=None, activation_fn=afn, max_epochs=10000,
									  validation_frequency=1000, lambda_=l, mini_batch_size=-1, W_init=wi, optimiser='gd', **gd_params)

							mlp.fit(X, y)

							curr_acc = accuracy_score(y, mlp.predict(X))
							if (curr_acc > best_acc):
								best_acc = curr_acc
								best_config = [sr, hh, l, afn, wi, mbs]
								print('\tAccuracy: {}; [config: eta={}; layer_size={}; lambda={}; activation={}; weight_initialisation={}; mini_batch_size={}]'.format(sr, hh, l, afn, wi, mbs))

							'''
							views = mlp.best_weights()
							W1 = views[0]
							b1 = views[1]
							W2 = views[2]
							b2 = views[3]

							#### PLOT DECISION BOUNDARY
							# plot the resulting classifier
							h = 0.02
							x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
							y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
							xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
												 np.arange(y_min, y_max, h))
							Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W1) + b1), W2) + b2
							Z = np.argmax(Z, axis=1)
							Z = Z.reshape(xx.shape)
							fig = plt.figure()
							plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
							plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
							plt.xlim(xx.min(), xx.max())
							plt.ylim(yy.min(), yy.max())
							fig.savefig('/Users/thomas/DevSandbox/InfiniteSandbox/tag-lab/abgrund/test/dump/mlp_decision_boundary_eta-{}_hidden-{}_lambda-{}_activation-{}_init-{}.png'.format(sr, hh, l, afn, wi))
							####
							'''
							print('-----------------------------------------------------------------------------')
	print('FINAL BEST ACC: {}, best config='.format(best_acc, *best_config))
	#'''

if (__name__ == '__main__'):
	test_mlp_toy_dataset()