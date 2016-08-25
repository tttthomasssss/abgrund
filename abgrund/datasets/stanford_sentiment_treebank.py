__author__ = 'thomas'
import os

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


def create_train_test_dev_split(dataset_path, fine_grained=False, lowercase=True, use_phrase_labels=False):
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

	# Add phrase level labelled sentences to training set
	if (use_phrase_labels):
		for text, label in phrase_label_dict.items():
			if (fine_grained):
				data_dict['1'][0].append(text.split())
				data_dict['1'][1].append(map_fine_grained_label(label))
			elif (not fine_grained and ((label > 0 and label <= 0.4) or (label > 0.6 and label <= 1.0))): # Binary doesn't include the neutral reviews
				data_dict['1'][0].append(text.split())
				data_dict['1'][1].append(map_binary_label(label))

	return data_dict['1'], data_dict['2'], data_dict['3']