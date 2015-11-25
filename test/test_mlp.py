__author__ = 'thomas'
from matplotlib import pyplot as plt
import numpy as np

from mlp.feedforward import MLP


def test_mlp_toy_dataset():
	N = 100 # Number of Samples per Class # 100000 for testing SGD, 100 for Batch GD
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

	# eta=1.0; layer_size=200; lambda=0.1; activation=relu; weight_initialisation=xavier (batch mode) ###
	# eta=0.1; layer_size=200; lambda=0.1; activation=relu; weight_initialisation=xavier; mini_batch_size=10000 (mini-batch mode) ###
	#'''
	gd_params = {
			'step_rate': 1.0
	}

	mlp = MLP(shape=[(2, 200), 200, (200, 3), 3], dropout_proba=None, activation_fn='relu', max_epochs=10000, gradient_check=False,
								  validation_frequency=1000, lambda_=0.1, mini_batch_size=-1, W_init='xavier', optimiser='gd', **gd_params)

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
	#'''

if (__name__ == '__main__'):
	test_mlp_toy_dataset()