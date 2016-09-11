__author__ = 'thomas'
from argparse import ArgumentParser
import csv
import os

from gensim.models import Word2Vec
from glove import Glove
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
import numpy as np

from abgrund.base import utils
from abgrund.datasets import stanford_sentiment_treebank as sts
from abgrund.models.mlp import MLP
from abgrund.vector_space.random_vectors import RandomVectorSpaceModel
from abgrund.vector_space.vsm import VectorSpaceModel

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

parser = ArgumentParser()
parser.add_argument('-cn', '--config-name', type=str, required=True, help='name of configuration')
parser.add_argument('-vf', '--vector-file', type=str, help='input vector file')
parser.add_argument('-vp', '--vector-path', type=str, help='path to input vector file')
parser.add_argument('-dp', '--dataset-path', type=str, help='path to dataset')
parser.add_argument('-df', '--dataset-file', type=str, help='dataset file')
parser.add_argument('-ef', '--experiment-file', type=str, help='experiment file')

ex = Experiment('iyyer_et_al_2015')
ex.observers.append(MongoObserver.create(db_name='Iyyer_et_al_2015'))

@ex.config
def config():
	config_name = ''
	vsm_type = ''
	vector_dim = -1
	vector_file = ''
	use_phrase_labels = False
	fine_grained = False
	p_keep_word = -1.0
	regularisation = ''
	lambda_ = 0.0
	mini_batch_size = 0
	optimiser = ''
	eta = -1.0
	max_epochs = -1
	composition = None
	layers = 0
	activation_function = ''


@ex.main
def run(config_name, vsm_type, vector_dim, vector_file, use_phrase_labels, fine_grained, p_keep_word, regularisation,
		lambda_, mini_batch_size, optimiser, eta, max_epochs, composition, layers, activation_function):

	# Split the dataset
	train_data, test_data, dev_data = sts.create_train_test_dev_split(dataset_path=args.dataset_path, fine_grained=fine_grained,
																	  lowercase=True, use_phrase_labels=use_phrase_labels)

	# Load the word2vec vectors
	#word2vec = Word2Vec.load_word2vec_format('/Users/thomas/DevSandbox/EpicDataShelf/tag-lab/wikipedia/word2vectors/word2vec_skip-gram_50_0.000100_10.bin', binary=True)
	#word2vec = Word2Vec.load_word2vec_format('/Users/thomas/DevSandbox/EpicDataShelf/tag-lab/amazon_reviews/word2vec_cbow_50_0.000100_5_5.bin', binary=True)
	#p_keep_word = 1.0
	#vsm = VectorSpaceModel(vsm=word2vec, vector_shape=word2vec.vector_size, vsm_type='word2vec', min_threshold=0)

	# Load the GloVe vectors
	#p = os.path.join(args.vector_path, args.vector_file)
	#glove = Glove.load_stanford(p)

	# Initialise the vector space with GloVe vectors
	#p_keep_word = 0.7
	#vsm = VectorSpaceModel(vsm=glove, vector_shape=glove.no_components, vsm_type='glove')

	# Initialise the vector space with random vectors
	p_keep_word = 1.
	rnd_vecs = RandomVectorSpaceModel(ndim=int(vector_dim))
	rnd_vecs.construct(train_data[0], initialise_immediately=True)
	vsm = VectorSpaceModel(vsm=rnd_vecs, vector_shape=rnd_vecs.dimensionality(), vsm_type='random')

	# Transform the dataset
	X_train = vsm.transform(train_data[0], composition=getattr(np, composition), p_keep_word=float(p_keep_word))
	y_train = np.array(train_data[1])
	X_test = vsm.transform(test_data[0], composition=getattr(np, composition))
	y_test = np.array(test_data[1])
	X_dev = vsm.transform(dev_data[0], composition=getattr(np, composition))
	y_dev = np.array(dev_data[1])

	### SHUFFLE THE LABELS AS A RANDOM TEST ###
	#y_train = shuffle(y_train)
	#y_train = np.zeros((y_train.shape), dtype=np.int16)
	#y_train[0] = 1
	#y_train[1] = 2
	#y_train[2] = 3
	#y_train[3] = 4
	#from sklearn.datasets import make_classification
	#_, y = make_classification(n_samples=y_test.shape[0], n_classes=5, n_features=300, n_informative=200)
	#y_test_shuffled = shuffle(y_test)
	#diff = y - y_test
	#print('{} out of {} (={}) kept the same label!'.format(np.where(diff==0)[0].shape[0], y.shape[0], np.where(diff==0)[0].shape[0] / y.shape[0]))
	#y_test = y

	# Learn the model
	#mlp = MLP(shape=[(50, 5), 5], dropout_proba=None, activation_fn='relu', max_epochs=100,
	 #         validation_frequency=10, optimiser='gd', optimiser_kwargs={'eta': 0.1})
	#mlp = MLP(shape=[(50, 50), 50, (50, 5), 5], dropout_proba=None, activation_fn='relu', max_epochs=500,
	 #         validation_frequency=10, optimiser='gd', optimiser_kwargs={'eta': 0.1}) # BASED ON BATCH GD
	#mlp = MLP(shape=[(50, 50), 50, (50, 5), 5], dropout_proba=None, activation_fn='tanh', max_epochs=100,
	 #         validation_frequency=10, optimiser='gd', optimiser_kwargs={'eta': 0.01})
	#mlp = MLP(shape=[(50, 50), 50, (50, 50), 50, (50, 5), 5], dropout_proba=None, activation_fn='relu', max_epochs=500,
	 #         validation_frequency=10, optimiser='gd', optimiser_kwargs={'eta': 0.1})
	#mlp = MLP(shape=[(50, 50), 50, (50, 50), 50, (50, 50), 50, (50, 5), 5], dropout_proba=None, activation_fn='relu', max_epochs=1500,
	 #         validation_frequency=10, optimiser='gd', optimiser_kwargs={'eta': 0.1})

	# Binary
	#mlp = MLP(shape=[(50, 50), 50, (50, 2), 2], dropout_proba=None, activation_fn='relu', max_epochs=100,
	 #         validation_frequency=10, optimiser='gd', optimiser_kwargs={'eta': 10000}, mini_batch_size=50)

	#mlp = MLP(shape=[(50, 2), 2], dropout_proba=None, activation_fn='sigmoid', max_epochs=100,
	 #         validation_frequency=10, optimiser='gd', optimiser_kwargs={'eta': 0.1}, mini_batch_size=50)

	#opt_kwargs = {'eta': eta, 'noise_eta': 0.01, 'mu': 0.99, 'momentum': 'standard'}
	opt_kwargs = {'eta': eta}
	shape = []
	for i in range(layers-1):
		shape.append((vector_dim, vector_dim)) # Hidden Layers
		shape.append(vector_dim) # Hidden Layer Bias

	shape.append((vector_dim, 5 if fine_grained else 2)) # Prediction Layer
	shape.append(5 if fine_grained else 2) # Prediction Layer Bias

	print('Running MLP with shape={}...'.format(shape))

	mlp = MLP(shape=shape, activation_fn=activation_function, max_epochs=max_epochs, optimiser=optimiser,
			  optimiser_kwargs=opt_kwargs, mini_batch_size=mini_batch_size,
			  regularisation=regularisation, lambda_=lambda_, shuffle=False, shuffle_mini_batches=True)

	mlp.fit(X_train, y_train, X_dev, y_dev)
	y_pred = mlp.predict(X_test)

	acc = accuracy_score(y_test, y_pred)
	print('Accuracy: {}'.format(acc))
	print('----------------------------------------------------')


	svm = LinearSVC()
	svm.fit(X_train, y_train)
	y_pred = svm.predict(X_test)
	print('SVM Accuracy: {}'.format(accuracy_score(y_test, y_pred)))

	utils.plot_learning_curve(mlp.loss_history_training_, '/Users/thomas/DevSandbox/InfiniteSandbox/tag-lab/Abgrund/test/dump/training_loss')
	utils.plot_learning_curve(mlp.loss_history_validation_, '/Users/thomas/DevSandbox/InfiniteSandbox/tag-lab/Abgrund/test/dump/validation_loss')

	#print('Gradient checking...')
	#diffs, errs = mlp._gradient_check(X_dev, y_dev)
	#print('Diffs: {}; Errors: {}'.format(diffs, errs))

	return acc

if (__name__ == '__main__'):
	args = parser.parse_args()

	# Load experiment id file
	with open(os.path.join(PROJECT_PATH, 'resources', 'parameter_optimisation', args.experiment_file), 'r') as csv_file:
		csv_reader = csv.reader(csv_file)
		experiments = []

		for line in csv_reader:
			experiments.append(line)

	for idx, (vsm_type, vector_dim, vector_file, use_phrase_labels, fine_grained, p_keep_word, regularisation, lambda_,
		mini_batch_size, eta, optimiser, max_epochs, composition, layers, activation_function) in enumerate(experiments, 1):

		print('Running experiment {} of {}...'.format(idx, len(experiments)))

		config_dict = {
			'config_name': args.config_name,
			'vsm_type': vsm_type,
			'vector_dim': int(vector_dim),
			'vector_file': vector_file,
			'use_phrase_labels': use_phrase_labels=='True',
			'fine_grained': fine_grained=='True',
			'p_keep_word': float(p_keep_word),
			'regularisation': regularisation,
			'lambda_': float(lambda_),
			'mini_batch_size': int(mini_batch_size),
			'optimiser': optimiser,
			'eta': float(eta),
			'max_epochs': int(max_epochs),
			'composition': composition,
			'layers': int(layers),
			'activation_function': activation_function
		}

		ex.run(config_updates=config_dict)