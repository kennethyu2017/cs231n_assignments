import numpy as np

"""
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning rate,
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
  """
  Performs vanilla stochastic gradient descent.

  config format:
  - learning_rate: Scalar learning rate.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)

  w -= config['learning_rate'] * dw
  return w, config


def sgd_momentum(w, dw, config=None):
	"""
	Performs stochastic gradient descent with momentum.

	config format:
	- learning_rate: Scalar learning rate.
	- momentum: Scalar between 0 and 1 giving the momentum value.
		Setting momentum = 0 reduces to sgd.
	- velocity: A numpy array of the same shape as w and dw used to store a moving
		average of the gradients.
	"""
	if config is None: config = {}
	config.setdefault('learning_rate', 1e-2)
	config.setdefault('momentum', 0.9)
	v = config.get('velocity', np.zeros_like(w))

	next_w = None
	#############################################################################
	# TODO: Implement the momentum update formula. Store the updated value in   #
	# the next_w variable. You should also use and update the velocity v.       #
	#############################################################################
	pass
	#kenneth yu
	mu = config.get('momentum')
	l_r = config.get('learning_rate')
	v = mu * v - l_r * dw
	next_w = w + v

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	# cache the v for each w.
	config['velocity'] = v

	return next_w, config



def rmsprop(x, dx, config=None):
	"""
  Uses the RMSProp update rule, which uses a moving average of squared gradient
  values to set adaptive per-parameter learning rates.

  config format:
  - learning_rate: Scalar learning rate.
  - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
    gradient cache.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - cache: Moving average of second moments of gradients.
	"""
	if config is None: config = {}
	config.setdefault('learning_rate', 1e-2)
	config.setdefault('decay_rate', 0.99)
	config.setdefault('epsilon', 1e-8)
	config.setdefault('cache', np.zeros_like(x))

	next_x = None
	#############################################################################
	# TODO: Implement the RMSprop update formula, storing the next value of x   #
	# in the next_x variable. Don't forget to update cache value stored in      #
	# config['cache'].                                                          #
	#############################################################################
	pass
	# kenneth yu
	cache = config['cache']
	decay_rate = config['decay_rate']
	l_r = config['learning_rate']
	eps = config['epsilon']
	cache = decay_rate * cache + (1 - decay_rate) * dx ** 2
	next_x = x + (-l_r) * dx / (np.sqrt(cache) + eps)

	#cache the cache for each x
	config['cache'] = cache

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return next_x, config


def adam(x, dx, config=None):
	"""
	Uses the Adam update rule, which incorporates moving averages of both the
	gradient and its square and a bias correction term.

	config format:
	- learning_rate: Scalar learning rate.
	- beta1: Decay rate for moving average of first moment of gradient.
	- beta2: Decay rate for moving average of second moment of gradient.
	- epsilon: Small scalar used for smoothing to avoid dividing by zero.
	- m: Moving average of gradient.
	- v: Moving average of squared gradient.
	- t: Iteration number.
	"""
	if config is None: config = {}
	config.setdefault('learning_rate', 1e-3)
	config.setdefault('beta1', 0.9)
	config.setdefault('beta2', 0.999)
	config.setdefault('epsilon', 1e-8)
	config.setdefault('m', np.zeros_like(x))
	config.setdefault('v', np.zeros_like(x))
	config.setdefault('t', 0)

	next_x = None
	#############################################################################
	# TODO: Implement the Adam update formula, storing the next value of x in   #
	# the next_x variable. Don't forget to update the m, v, and t variables     #
	# stored in config.                                                         #
	#############################################################################
	pass
	#	kenneth yu
	beta1 = config['beta1']
	beta2 = config['beta2']
	l_r = config['learning_rate']
	eps = config['epsilon']
	m = config['m']
	v = config['v']
	t = config['t']

	m = beta1 * m + (1 - beta1) * dx
	v = beta2 * v + (1 - beta2)	* (dx ** 2)

	# !!!!! -NOTE- must use a deep copy to diction config.dict just save the reference
	# of m,v. otherwise, even if we cache the m, v value here,
	# the following warm-up step will change the cached value of m,v in config. and we should not cache the
	#	warm up value of m,v in config.
	config['m'] = m.copy()
	config['v']	= v.copy()


	#	warm-up m,v , which are tiny in the first few iterations.
	# if t < 10:
	t = t+1
	m /= (1 - beta1 ** t)       	#in-place assignment.
	v	/= (1 - beta2 ** t)         #in-place assignment.
	#	cache them.
	config['t']	=	t

	# use the warm up m, v
	next_x = x + (-l_r) * m/(np.sqrt(v) + eps)


	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return next_x, config





