import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
	"""
	A three-layer convolutional network with the following architecture:

	conv - relu - 2x2 max pool - affine - relu - affine - softmax
	3 layer:  [conv] -> [FC] -> [output]

	The network operates on minibatches of data that have shape (N, C, H, W)
	consisting of N images, each with height H and width W and with C input
	channels.
	"""

	def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
							 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
							 dtype=np.float32):
		"""
		Initialize a new network.

		Inputs:
		- input_dim: Tuple (C, H, W) giving size of input data
		- num_filters: Number of filters to use in the convolutional layer
		- filter_size: Size of filters to use in the convolutional layer
		- hidden_dim: Number of units to use in the fully-connected hidden layer
		- num_classes: Number of scores to produce from the final affine layer.
		- weight_scale: Scalar giving standard deviation for random initialization
			of weights.
		- reg: Scalar giving L2 regularization strength
		- dtype: numpy datatype to use for computation.
		"""
		self.params = {}
		self.reg = reg
		self.dtype = dtype

		############################################################################
		# TODO: Initialize weights and biases for the three-layer convolutional    #
		# network. Weights should be initialized from a Gaussian with standard     #
		# deviation equal to weight_scale; biases should be initialized to zero.   #
		# All weights and biases should be stored in the dictionary self.params.   #
		# Store weights and biases for the convolutional layer using the keys 'W1' #
		# and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
		# hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
		# of the output affine layer.                                              #
		############################################################################
		pass
		#	kenneth yu
		# 3 layer:  [conv] -> [FC] -> [output]
		# conv - relu - 2x2 max pool - hidden_affine - relu - output_affine - softmax
		# the parameters to be inited:
		#	conv: w, b;		relu: none;   max: none;   affine: w, b;  output:
		(C, H, W) = input_dim
		conv_w_shape = (num_filters, C, filter_size,filter_size)
		conv_b_shape = (num_filters,)
		#	after conv , the img should keep the same spatial size.after 2x2 maxpool half the spatial size.
		# conv layer output shape (F, H', W') for each data
		conv_output_shape = (num_filters, H/2, W/2)

		#	see hidden_dim as the # of filters in hidden affine layer.
		# each filter(Neuron) can take care of all the previous layer activation maps.
		hidden_affine_w_shape = (hidden_dim,) + conv_output_shape
		hidden_affine_b_shape =	(hidden_dim,)  # a tuple.

		# see the output of hidden affine layer as hidden_dim *(1*1) activation maps.
		hidden_affine_output_shape = (hidden_dim, 1, 1)
		#	see num_classes the # of filters in output affine layer.
		output_affine_w_shape	=	(num_classes, ) + hidden_affine_output_shape
		output_affine_b_shape	=	(num_classes,)
		### the final score out should have shape( num_class, 1, 1)

		# init_weights = lambda i_dim,o_dim: np.random.randn(i_dim, o_dim).astype(dtype)\
		# 	 																	 * np.sqrt(2.0/(i_dim))

		init_weights = lambda dim: weight_scale * np.random.randn(*dim).astype(dtype)

		# bias init to zero. shape(1, o_dim)
		init_bias = lambda dim: np.zeros(dim, dtype=dtype)

		#	1. conv layer. params: W1,b1
		#	w shape: (F, C, HH, HW). b shape: (F,)
		self.params['W1'] = init_weights(conv_w_shape)
		self.params['b1']	= init_bias(conv_b_shape)

		#	2. hidden_affine. params: W2, b2
		self.params['W2'] = init_weights(hidden_affine_w_shape)
		self.params['b2'] = init_bias(hidden_affine_b_shape)

		#	3. output_affine. params: W3,b3
		self.params['W3'] = init_weights(output_affine_w_shape)
		self.params['b3'] = init_bias(output_affine_b_shape)

		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################

		for k, v in self.params.iteritems():
			self.params[k] = v.astype(dtype)
			print '--> init param name:',k, ' shape:', v.shape


	def loss(self, X, y=None, layer_verbose=False):
		"""
		Evaluate loss and gradient for the three-layer convolutional network.

		Input / output: Same API as TwoLayerNet in fc_net.py.
		"""
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		W3, b3 = self.params['W3'], self.params['b3']

		# pass conv_param to the forward pass for the convolutional layer
		filter_size = W1.shape[2]
		conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

		# pass pool_param to the forward pass for the max-pooling layer
		pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

		scores = None
		############################################################################
		# TODO: Implement the forward pass for the three-layer convolutional net,  #
		# computing the class scores for X and storing them in the scores          #
		# variable.                                                                #
		############################################################################
		pass
		#	kenneth yu

		#	init
		out = []
		cache = []
		ca = []

		#	fp ==> !!! take care of the dims of each out !!!!!
		# conv - relu - 2x2 max pool - affine - relu - affine - softmax
		# 3 layers:  [conv] -> [fc] -> [output]
		#	conv layer
		out, ca = conv_relu_pool_forward(X, W1, b1, conv_param,pool_param)
		cache.append(ca)
		# print ' @@ conv out shape,', out.shape

		#	use conv to replace the affine forward fc calculation.
		#	input for conv_relu_forward:
		# -N, C, H, W = x.shape
	  # -F, _, HH, WW = w.shape
  	# -stride, pad = conv_param['stride'], conv_param['pad']
		out, ca = conv_relu_forward(out, W2, b2, {'stride':1,'pad':0})
		cache.append(ca)
		# print ' @@ hidden affine out shape,', out.shape

		#	last output layer. use conv to replace the affine forward fc.
		out, ca = conv_forward_fast(out, W3, b3, {'stride':1, 'pad':0})
		cache.append(ca)
		# print ' @@ output affine out shape,', out.shape

		scores = out			# shape (N, C, 1, 1)
		# print ' @@ scores shape,', scores.shape

		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################

		if y is None:
			return scores

		loss, grads = 0, {}
		############################################################################
		# TODO: Implement the backward pass for the three-layer convolutional net, #
		# storing the loss and gradients in the loss and grads variables. Compute  #
		# data loss using softmax, and make sure that grads[k] holds the gradients #
		# for self.params[k]. Don't forget to add L2 regularization!               #
		############################################################################
		pass
		#	kenneth yu
		#	calc data loss. take care of the dims of scores (N,C,1,1).
		data_loss, d_scores = softmax_loss(scores.squeeze(), y)
		d_scores = d_scores.reshape(scores.shape)

		#	calc reg loss
		l2_norm = lambda w: (np.linalg.norm(w.ravel()))**2

		reg_loss = (1.0/2) * self.reg * (l2_norm(W1) + l2_norm(W2) + l2_norm(W3))

		#	full loss
		loss = data_loss + reg_loss

		####### calc grads	#####

		#	data grads.
		#	last output layer
		ca = cache.pop()
		dx, grads['W3'], grads['b3'] = conv_backward_fast(d_scores, ca)

		#	hidden affine layer
		ca = cache.pop()
		dx, grads['W2'], grads['b2'] = conv_relu_backward(dx,ca)

		#	conv layer
		ca = cache.pop()
		dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, ca)

		#	add the reg grads.ignore bias.
		grads['W1'] += self.reg * W1
		grads['W2'] += self.reg * W2
		grads['W3'] += self.reg * W3
		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################
		return loss, grads


pass
