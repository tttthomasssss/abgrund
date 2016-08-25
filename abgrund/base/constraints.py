__author__ = 'thomas'
import numpy as np


def max_weight_norm(weights, max_norm):
	norm_weights = []
	for W in weights:
		curr_norm = np.linalg.norm(W)
		if (curr_norm > max_norm):
			norm_weights.append(W * (max_norm / curr_norm))
	return norm_weights