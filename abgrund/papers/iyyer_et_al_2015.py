__author__ = 'thomas'
from argparse import ArgumentParser
import os

from glove import Glove
from sklearn.metrics import accuracy_score
import numpy as np

from abgrund.models.mlp import MLP
from abgrund.vector_space.random_vectors import RandomVectorSpaceModel
from abgrund.vector_space.vsm import VectorSpaceModel

parser = ArgumentParser()
parser.add_argument('-vf', '--vector-file', type=str, help='input vector file')
parser.add_argument('-vp', '--vector-path', type=str, help='path to input vector file')
parser.add_argument('-dp', '--dataset-path', type=str, help='path to dataset')
parser.add_argument('-df', '--dataset-file', type=str, help='dataset file')

# TODO: AdaGrad, use all data

def map_fine_grained_label(label):
    sentiment = -1
    if (label >= 0 and label <= 0.2):
        sentiment = 0
    elif (label > 0.2 and label <= 0.4):
        sentiment = 1
    elif (label > 0.4 and label <= 0.6):
        sentiment = 2
    elif (label > 0.6 and label <= 0.8):
        sentiment = 3
    elif (label > 0.8 and label <= 1.0):
        sentiment = 4

    return sentiment


def map_binary_label(label):
    sentiment = -1
    if (label >= 0 and label <= 0.4):
        sentiment = 0
    elif (label > 0.6 and label <= 1.0):
        sentiment = 1

    return sentiment


def create_train_test_dev_split_sts(dataset_path, fine_grained=False, lowercase=True):
    split_file = 'datasetSplit.txt'
    data_file = 'datasetSentences.txt'
    dictionary_file = 'dictionary.txt'
    labels_file = 'sentiment_labels.txt'

    data_dict = {
        '1': ([], []), # train
        '2': ([], []), # test
        '3': ([], []) # dev
    }

    # many mangled chars in sentences (datasetSentences.txt)
    chars_sst_mangled = ['à', 'á', 'â', 'ã', 'æ', 'ç', 'è', 'é', 'í',
                         'í', 'ï', 'ñ', 'ó', 'ô', 'ö', 'û', 'ü']
    sentence_fixups = [(char.encode('utf-8').decode('latin1'), char) for char in chars_sst_mangled]
    # more junk, and the replace necessary for sentence-phrase consistency
    sentence_fixups.extend([
        ('Â', ''),
        ('\xa0', ' '),
        ('-LRB-', '('),
        ('-RRB-', ')'),
    ])

    # Build phrase and label dicts
    phrase_dict = {}
    label_dict = {}
    phrase_label_dict = {}
    with open(os.path.join(dataset_path, dictionary_file), 'r') as dict_file, open(os.path.join(dataset_path, labels_file), 'r') as f_labels:
        next(f_labels) # skip first line in labels file

        for data_line, labels_line in zip(dict_file, f_labels):
            text, id = data_line.strip().split('|')
            iid, label = labels_line.strip().split('|')
            for junk, fix in sentence_fixups:
                text = text.replace(junk, fix)

            if (lowercase):
                text = text.lower()

            phrase_dict[text] = int(id)
            label_dict[iid] = float(label)

            phrase_label_dict[text] = float(label)

    # Read sentences
    with open(os.path.join(dataset_path, split_file), 'r') as f_split, open(os.path.join(dataset_path, data_file), 'r') as f_data:
        next(f_split) # skip first line
        next(f_data)
        for split_line, data_line in zip(f_split, f_data):
            key = split_line.strip().split(',')[1]
            text = data_line.strip().split('\t')[1]
            for junk, fix in sentence_fixups:
                text = text.replace(junk, fix)

            if (lowercase):
                text = text.lower()

            label = phrase_label_dict[text]

            if (fine_grained):
                data_dict[key][0].append(text.split())
                data_dict[key][1].append(map_fine_grained_label(label))
            elif (not fine_grained and ((label > 0 and label <= 0.4) or (label > 0.6 and label <= 1.0))): # Binary doesn't include the neutral reviews
                data_dict[key][0].append(text.split())
                data_dict[key][1].append(map_binary_label(label))

    return data_dict['1'], data_dict['2'], data_dict['3']


if (__name__ == '__main__'):
    args = parser.parse_args()

    # Split the dataset
    train_data, test_data, dev_data = create_train_test_dev_split_sts(dataset_path=args.dataset_path, fine_grained=False,
                                                                      lowercase=True)

    # Load the GloVe vectors
    #p = os.path.join(args.vector_path, args.vector_file)
    #glove = Glove.load_stanford(p)

    # Initialise the vector space with GloVe vectors
    #p_keep_word = 0.5
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

    mlp = MLP(shape=[(50, 50), 50, (50, 2), 2], dropout_proba=None, activation_fn='relu', max_epochs=100,
              validation_frequency=10, optimiser='gd', optimiser_kwargs={'eta': 0.1}, mini_batch_size=50)

    mlp.fit(X_train, y_train, X_dev, y_dev)
    y_pred = mlp.predict(X_test)

    print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))