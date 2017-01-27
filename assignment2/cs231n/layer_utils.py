from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  forward = affine_forward(x, w, b)
  a, fc_cache = forward
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


pass


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  fast = conv_forward_fast(x, w, b, conv_param)
  a, conv_cache = fast
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


#	kenneth yu.
def affine_bn_relu_forward(x, **kwargs):
  """
  Convenience layer that perorms an affine + BN followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - gamma, beta, bn_param: transform param for BN layer


  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  # ** kwargs:  w, b, gamma, beta, bn_param
  W = kwargs.pop('W')
  b = kwargs.pop('b')
  gamma = kwargs.pop('gamma')
  beta = kwargs.pop('beta')
  bn_param = kwargs.pop('bn_param')

  #	affine
  af_out, af_cache = affine_forward(x, W, b)

  # print '== out of affine: ', af_out

  # bn
  bn_out, bn_cache = batchnorm_forward(af_out, gamma, beta, bn_param)

  # print '== out of bn: ', bn_out

  # relu
  relu_out, relu_cache = relu_forward(bn_out)

  # print '== out of ReLU: ', relu_out

  cache = (af_cache, bn_cache, relu_cache)

  return relu_out, cache

# kenneth yu
def affine_bn_relu_backward(dout, cache):
	"""
	Backward pass for the affine-bn-relu convenience layer
	return:
		- d_x
		- gradients for BN layer:d_gamma, d_beta
		- gradints for affine layer: d_w, d_b
	"""
	af_cache, bn_cache, relu_cache = cache

	#	relu bp
	d_relu = relu_backward(dout, relu_cache)

	# bn bp
	d_bn, d_gamma, d_beta = batchnorm_backward_alt(d_relu, bn_cache)

	# af bp
	d_x, d_w, d_b = affine_backward(d_bn, af_cache)

	return d_x,d_w, d_b, d_gamma, d_beta

