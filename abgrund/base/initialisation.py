__author__ = 'thomas'
import math
import numpy as np


def randomise_normal(shape, loc=0, scale=1, random_state=None):
	rnd = np.random if random_state is None else random_state

	weights = []

	for s in shape:
		weights.append(rnd.normal(loc, scale, s))

	return weights

# Check http://deeplearning.net/tutorial/mlp.html#weight-initialization
def randomise_uniform_sigmoid(shape, random_state=None):
	rnd = np.random if random_state is None else random_state

	weights = []

	for s in shape:
		# Determine fan_in & fan_out
		if (isinstance(s, tuple)):
			fan_out = s[1]
			fan_in = s[0]
		else:
			fan_out = 1
			fan_in = s

		xavier = math.sqrt(6) / math.sqrt(fan_in + fan_out)
		weights.append(rnd.uniform(-4 * xavier, 4 * xavier, size=s))

	return weights


def randomise_uniform_tanh(shape, random_state=None):
	rnd = np.random if random_state is None else random_state

	weights = []

	for s in shape:
		# Determine fan_in & fan_out
		if (isinstance(s, tuple)):
			fan_out = s[1]
			fan_in = s[0]
		else:
			fan_out = 1
			fan_in = s

		xavier = math.sqrt(6) / math.sqrt(fan_in + fan_out)
		weights.append(rnd.uniform(-xavier, xavier, size=s))

	return weights


def randomise_uniform_relu(shape, random_state=None):
	rnd = np.random if random_state is None else random_state

	weights = []

	for s in shape:
		# Determine fan_in & fan_out
		if (isinstance(s, tuple)):
			fan_out = s[1]
			fan_in = s[0]
		else:
			fan_out = 1
			fan_in = s

		xavier = math.sqrt(6) / math.sqrt(fan_in + fan_out)
		weights.append(rnd.uniform(-xavier, xavier, size=s))

	return weights


def randn(shape, random_state=None, scale=0.01):
	rnd = np.random if random_state is None else random_state

	weights = []

	for s in shape:
		weights.append(scale * rnd.randn(*s) if isinstance(s, tuple) else rnd.randn(s))

	return weights


def randomise_normal_relu(shape, random_state=None):
	return randomise_normal(shape, 0, 0.01, random_state)