__author__ = 'thomas'
from argparse import ArgumentParser
import os

from gensim.models import Word2Vec
from glove import Glove
from sklearn.metrics import accuracy_score
import numpy as np

from abgrund.datasets import stanford_sentiment_treebank as sts
from abgrund.models.mlp import MLP
from abgrund.vector_space.random_vectors import RandomVectorSpaceModel
from abgrund.vector_space.vsm import VectorSpaceModel

parser = ArgumentParser()
parser.add_argument('-vf', '--vector-file', type=str, help='input vector file')
parser.add_argument('-vp', '--vector-path', type=str, help='path to input vector file')
parser.add_argument('-dp', '--dataset-path', type=str, help='path to dataset')
parser.add_argument('-df', '--dataset-file', type=str, help='dataset file')

# TODO: AdaGrad

if (__name__ == '__main__'):
    args = parser.parse_args()

    # Split the dataset
    train_data, test_data, dev_data = sts.create_train_test_dev_split(dataset_path=args.dataset_path, fine_grained=True,
                                                                      lowercase=True, use_phrase_labels=True)

    # Load the word2vec vectors
    #word2vec = Word2Vec.load_word2vec_format('/Users/thomas/DevSandbox/EpicDataShelf/tag-lab/wikipedia/word2vectors/word2vec_skip-gram_50_0.000100_10.bin', binary=True)
    #p_keep_word = 0.7
    #vsm = VectorSpaceModel(vsm=word2vec, vector_shape=word2vec.vector_size, vsm_type='word2vec')

    # Load the GloVe vectors
    #p = os.path.join(args.vector_path, args.vector_file)
    #glove = Glove.load_stanford(p)

    # Initialise the vector space with GloVe vectors
    #p_keep_word = 0.7
    #vsm = VectorSpaceModel(vsm=glove, vector_shape=glove.no_components, vsm_type='glove')

    # Initialise the vector space with random vectors
    p_keep_word = 0.5
    rnd_vecs = RandomVectorSpaceModel(ndim=50)
    rnd_vecs.construct(train_data[0], initialise_immediately=True)
    vsm = VectorSpaceModel(vsm=rnd_vecs, vector_shape=rnd_vecs.dimensionality(), vsm_type='random')


    # Transform the dataset
    X_train = vsm.transform(train_data[0], composition=np.average, p_keep_word=p_keep_word)
    y_train = np.array(train_data[1])
    X_test = vsm.transform(test_data[0], composition=np.average)
    y_test = np.array(test_data[1])
    X_dev = vsm.transform(dev_data[0], composition=np.average)
    y_dev = np.array(dev_data[1])

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

    mlp = MLP(shape=[(50, 50), 50, (50, 5), 5], activation_fn='relu', max_epochs=10, optimiser='adagrad',
              optimiser_kwargs={'eta': 0.1}, mini_batch_size=30)

    mlp.fit(X_train, y_train, X_dev, y_dev)
    y_pred = mlp.predict(X_test)

    print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))