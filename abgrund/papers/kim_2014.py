__author__ = 'thomas'
from argparse import ArgumentParser
import os

from abgrund.datasets import stanford_sentiment_treebank as sts
from abgrund.models.cnn import CNN
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

if (__name__ == '__main__'):
	args = parser.parse_args()

	# Split the dataset
	train_data, test_data, dev_data = sts.create_train_test_dev_split(dataset_path=args.dataset_path, fine_grained=True,
																	  lowercase=True, use_phrase_labels=False)

	# Initialise the vector space with random vectors
	rnd_vecs = RandomVectorSpaceModel(ndim=30)
	rnd_vecs.construct(train_data[0], initialise_immediately=True)
	vsm = VectorSpaceModel(vsm=rnd_vecs, vector_shape=rnd_vecs.dimensionality(), vsm_type='random')

	cnn = CNN(shape=[(200, 5), 5], activation_fn='relu', max_epochs=10, optimiser='gd', vector_space_model=vsm,
			  optimiser_kwargs={'eta': 0.1}, mini_batch_size=50, num_filters=100, filter_ngram_ranges=[2, 3])


	cnn._forward_propagation(train_data[0][0])