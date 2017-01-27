import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  pass
  #kenneth yu

  # reshape input to (N, D), and compute w*x + b
  # out = x.dot(w) + np.expand_dims(b,axis=1).T
  out = x.reshape(x.shape[0], -1).dot(w) + b  #shape (N, M)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)
    - b: bias, of shape(M, )

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  pass
  #kenneth yu. dx, dw, db should be double decision.float64.

  dx = dout.dot(w.T)   #shape (N,D)
  #reshape back to (N, d1,..., d_k)
  dx = dx.reshape(-1, *(x.shape[1:]))

  dw = x.reshape(x.shape[0], -1).T.dot(dout)  #shape (D,M)
  db = np.sum(dout, axis = 0)   #shape (M, )



  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
	"""
	Computes the forward pass for a layer of rectified linear units (ReLUs).

	Input:
	- x: Inputs, of any shape

	Returns a tuple of:
	- out: Output, of the same shape as x
	- cache: x
	"""
	out = None
	#############################################################################
	# TODO: Implement the ReLU forward pass.                                    #
	#############################################################################
	pass
	# kenneth yu. use scalar 0 to boadcast to shape of x.
	# calc how many cross kinks when num check grads. 1e-5 is step 'h'.
	if (1e-5) in x:
		print '+@@@@+ relu cross kinks -> ', np.sum(x==1e-5)

	out = np.maximum(0.0, x)



	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	cache = x
	return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  pass
  #kenneth yu
  dx = (x > np.float64(0.0)) * dout


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
	"""
	Forward pass for batch normalization.

	During training the sample mean and (uncorrected) sample variance are
	computed from minibatch statistics and used to normalize the incoming data.
	During training we also keep an exponentially decaying running mean of the mean
	and variance of each feature, and these averages are used to normalize data
	at test-time..

	At each timestep we update the running averages for mean and variance using
	an exponential decay based on the momentum parameter:

	running_mean = momentum * running_mean + (1 - momentum) * sample_mean
	running_var = momentum * running_var + (1 - momentum) * sample_var

	Note that the batch normalization paper suggests a different test-time
	behavior: they compute sample mean and variance for each feature using a
	large number of training images rather than using a running average. For
	this implementation we have chosen to use running averages instead since
	they do not require an additional estimation step; the torch7 implementation
	of batch normalization also uses running averages.

	Input:
	- x: Data of shape (N, D)
	- gamma: Scale parameter of shape (D,)
	- beta: Shift paremeter of shape (D,)
	- bn_param: Dictionary with the following keys:
		- mode: 'train' or 'test'; required
		- eps: Constant for numeric stability
		- momentum: Constant for running mean / variance.
		- running_mean: Array of shape (D,) giving running mean of features
		- running_var Array of shape (D,) giving running variance of features

	Returns a tuple of:
	- out: of shape (N, D)
	- cache: A tuple of values needed in the backward pass
	"""
	mode = bn_param['mode']

	# default 1e-05
	eps = bn_param.get('eps', 1e-5)

	# defalut 0.9
	momentum = bn_param.get('momentum', 0.9)

	N, D = x.shape
	#	initialize to zeros!!!
	running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
	running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

	out, cache = None, None
	if mode == 'train':
		#############################################################################
		# TODO: Implement the training-time forward pass for batch normalization.   #
		# Use minibatch statistics to compute the mean and variance, use these      #
		# statistics to normalize the incoming data, and scale and shift the        #
		# normalized data using gamma and beta.                                     #
		#                                                                           #
		# You should store the output in the variable out. Any intermediates that   #
		# you need for the backward pass should be stored in the cache variable.    #
		#                                                                           #
		# You should also use your computed sample mean and variance together with  #
		# the momentum variable to update the running mean and running variance,    #
		# storing your result in the running_mean and running_var variables.        #
		#############################################################################
		pass
		# kenneth yu
		# calc mean , var of this mini-batch data for each feature.
		mean = np.mean(x, axis=0, dtype=x.dtype)		# shape (D,).  should be (D,1)? ==> no need.can brdcast.
		var = np.var(x, axis=0, dtype=x.dtype)		#shape (D,)

		#	normalize input x per feature.
		x_hat = (x - mean) / np.sqrt(var + eps)	 # shape (N,D)

		#	linear transform together with gamma and beta. gamma and beta shape (D,).
		out = gamma * x_hat + beta			# shape (N,D)

		# store cache , update the running average.
		# memory view?? need copy??
		# memory view?? need copy?? if BP is followed immediately should be ok?
		cache = (x, x_hat, gamma, beta, mean, var, eps)

		#	at the initial steps, the running_mean and running_var are initialized to zero, so running_mean
		#	and running_var are close to 0, need 1/(1-momentum) steps to warm-up.
		running_mean = running_mean * momentum + (1-momentum) * mean
		running_var = running_var * momentum + (1-momentum) * var

		#	debug print. layer print each layer var and mean. compare on/off BN. do on the FCNetwork?

		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
	elif mode == 'test':
		#############################################################################
		# TODO: Implement the test-time forward pass for batch normalization. Use   #
		# the running mean and variance to normalize the incoming data, then scale  #
		# and shift the normalized data using gamma and beta. Store the result in   #
		# the out variable.                                                         #
		#############################################################################
		#
		pass
		#	kenneth yu
		#	normalize input x per feature together with fixed running average and var.
		x_hat = (x - running_mean) / np.sqrt(running_var + eps)	 # shape (N,D)

		#	linear transform together with learned gamma and beta. gamma and beta shape (D,).
		out = gamma * x_hat + beta			# shape (N,D)

		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
	else:
		raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

	# Store the updated running means back into bn_param
	bn_param['running_mean'] = running_mean
	bn_param['running_var'] = running_var

	return out, cache


def batchnorm_backward(dout, cache):
	"""
	Backward pass for batch normalization.

	For this implementation, you should write out a computation graph for
	batch normalization on paper and propagate gradients backward through
	intermediate nodes.

	Inputs:
	- dout: Upstream derivatives, of shape (N, D)
	- cache: Variable of intermediates from batchnorm_forward.

	Returns a tuple of:
	- dx: Gradient with respect to inputs x, of shape (N, D)
	- dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
	- dbeta: Gradient with respect to shift parameter beta, of shape (D,)
	"""
	dx, dgamma, dbeta = None, None, None
	#############################################################################
	# TODO: Implement the backward pass for batch normalization. Store the      #
	# results in the dx, dgamma, and dbeta variables.                           #
	#############################################################################
	pass
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
	"""
	Alternative backward pass for batch normalization.

	For this implementation you should work out the derivatives for the batch
	normalizaton backward pass on paper and simplify as much as possible. You
	should be able to derive a simple expression for the backward pass.

	Note: This implementation should expect to receive the same cache variable
	as batchnorm_backward, but might not use all of the values in the cache.

	Inputs / outputs: Same as batchnorm_backward
	"""
	dx, dgamma, dbeta = None, None, None
	#############################################################################
	# TODO: Implement the backward pass for batch normalization. Store the      #
	# results in the dx, dgamma, and dbeta variables.                           #
	#                                                                           #
	# After computing the gradient with respect to the centered(normalized)     #
	# inputs, youshould be able to compute gradients with respect to the inputs #
	# in a
	# single statement; our implementation fits on a single 80-character line.  #
	#############################################################################
	pass
	# kenneth yu
	# fetch cached data.
	#	cache = (x, x_hat, gamma, beta, mean, var, eps)
	(x, x_hat, gamma, beta, mean, var, eps) = cache
	#	dout is dyi. use chained rule.
	dx_hat = dout * gamma	 # shape (N,D)
	#	batch size
	m = x.shape[0]
	# element-wise calc
	s = np.sum(x,axis=0)  # shape (D,). sum each feature across all the batch data
	# dx = dx_hat * ( x_hat * (-1) * (1.0 / (var + eps)) * (1./m) * (x - (1./m) * s)
	#                 + ((m-1.0)/m) / np.sqrt(var + eps) )

	# dx = (-1.0/m) * dx_hat.sum(axis=0) * ( 1.0/np.sqrt(var+eps) + x/(var+eps)) +\
	# 		 (1.0/m) * np.sum(x_hat * dx_hat, axis=0)/(var+eps) + dx_hat / np.sqrt(var+eps)
	sq = np.sqrt(var + eps)
	# dx = (-1.0/m) / np.sqrt(var+eps) * dx_hat.sum(axis=0) \
	# 		 + (-1.0/m) / (var+eps) * (x - s/m) * np.sum(x_hat * dx_hat, axis=0) \
	# 		 + dx_hat / np.sqrt(var+eps)
	dx = (-1.0/m) / sq * dx_hat.sum(axis=0) \
			 + (-1.0/m) / (var+eps) * (x - s/m) * np.sum(x_hat * dx_hat, axis=0) \
			 + dx_hat / sq

	dgamma = (x_hat * dout).sum(axis=0)   # shape (D,)
	dbeta = dout.sum(axis=0)  # shape(D,)

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
	"""
	Performs the forward pass for (inverted) dropout.

	Inputs:
	- x: Input data, of any shape
	- dropout_param: A dictionary with the following keys:
		- p: Dropout parameter. We drop each neuron output with probability p.
		- mode: 'test' or 'train'. If the mode is train, then perform dropout;
			if the mode is test, then just return the input.
		- seed: Seed for the random number generator. Passing seed makes this
			function deterministic, which is needed for gradient checking but not in
			real networks.

	Outputs:
	- out: Array of the same shape as x.
	- cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
		mask that was used to multiply the input; in test mode, mask is None.
	"""
	p, mode = dropout_param['p'], dropout_param['mode']
	#	use seed to be deterministic. needed for gradient check . not in real networks.
	if 'seed' in dropout_param:
		np.random.seed(dropout_param['seed'])

	# print ' ~~~~ drop out fp seed: ', dropout_param['seed']

	mask = None
	out = None

	if mode == 'train':
		###########################################################################
		# TODO: Implement the training phase forward pass for inverted dropout.   #
		# Store the dropout mask in the mask variable.                            #
		###########################################################################
		pass
		# kenneth yu
		# preserve the 'units'  < p
		mask = ( np.random.rand(*x.shape).astype(np.float64) < p )
		#	cache mask belowing.
		#	check the mask:
		# print '== drop out mask : ', mask
		# print '== drop out mask number of True: ', np.sum(mask), ' data x size:', x.size, \
		# 	' cut-down unit ratio: %f' % np.mean(mask == False)
		# print ' == dropout mask True mean over each data sample: ', np.mean(mask, axis = 1)
		# print ' == > killed output when mask=True but corsp relu-out=zero', np.sum(mask & (x==0))
		# print ' size of shape ' ,mask.shape

		# inverted dropout. p control over all output dimensions and across one batch.
		out = (x * mask / np.float64(p)).astype(x.dtype,copy=False)
		# print ' == dropout out =>', out

		###########################################################################
		#                            END OF YOUR CODE                             #
		###########################################################################
	elif mode == 'test':
		###########################################################################
		# TODO: Implement the test phase forward pass for inverted dropout.       #
		###########################################################################
		pass
		#	kenneth yu
		#	during test, we dont do dropout. and weight scaling not needed due to
		# inverted dropout at FP.
		out = x

		###########################################################################
		#                            END OF YOUR CODE                             #
		###########################################################################

	cache = (dropout_param, mask)
	out = out.astype(x.dtype, copy=False)
	return out, cache


def dropout_backward(dout, cache):
	"""
	Perform the backward pass for (inverted) dropout.

	Inputs:
	- dout: Upstream derivatives, of any shape
	- cache: (dropout_param, mask) from dropout_forward.
	"""
	dropout_param, mask = cache
	mode = dropout_param['mode']

	dx = None
	if mode == 'train':
		###########################################################################
		# TODO: Implement the training phase backward pass for inverted dropout.  #
		###########################################################################
		pass
		#	kenneth yu
		# print ' $$$ drop out BP mask number of True: ', np.sum(mask),'cut-down unit ratio: %f' % np.mean(mask == False)
		#	inverted dropout , so dont forget the /p.
		dx = dout * mask / np.float64(dropout_param['p'])
		dx = dx.astype(dout.dtype)

		# print '@@ dx', dx
		# print '@@ dout', dout

		###########################################################################
		#                            END OF YOUR CODE                             #
		###########################################################################
	elif mode == 'test':
		dx = dout
	return dx


def conv_forward_naive(x, w, b, conv_param):
	"""
	A naive implementation of the forward pass for a convolutional layer.

	The input consists of N data points, each with C channels, height H and width
	W. We convolve each input with F different filters, where each filter spans
	all C channels and has height HH and width HH.

	Input:
	- x: Input data of shape (N, C, H, W)
	- w: Filter weights of shape (F, C, HH, WW)
	- b: Biases, of shape (F,)
	- conv_param: A dictionary with the following keys:
		- 'stride': The number of pixels between adjacent receptive fields in the
			horizontal and vertical directions.
		- 'pad': The number of pixels that will be used to zero-pad the input.

	Returns a tuple of:
	- out: Output data, of shape (N, F, H', W') where H' and W' are given by
		H' = 1 + (H + 2 * pad - HH) / stride
		W' = 1 + (W + 2 * pad - WW) / stride
	- cache: (x, w, b, conv_param)
	"""
	out = None
	#############################################################################
	# TODO: Implement the convolutional forward pass.                           #
	# Hint: you can use the function np.pad for padding.                        #
	#############################################################################
	pass
	#	kenneth yu
	#	get the shape  of data, and conv block. and parameter of conv.
	(N, C, H, W) = x.shape
	(F, C, HH, HW) = w.shape
	stride = conv_param.get('stride', 1)
	pad = conv_param.get('pad',0)

	#	sanity check:
	# print '==> conv_forward_naive -> x shape: ', x.shape
	# print '==> conv_forward_naive -> w shape: ', w.shape
	# print '==> conv_forward_naive -> b shape: ', b.shape
	# print '==> conv_forward_naive -> conv params: ', conv_param


	# zero-padding the data. np.pad is to pad the edge of each appointed axis.
	#	we pad along spatial axises only: H, W.
	# can use resize??? no.
	x = np.pad(x,((0,0),(0,0),(pad,pad), (pad,pad) ), 'constant', constant_values=0)

	# stretch the data block according to the conv block size and batch data numbers.
	# or use multi-dimension array dot?
	activation_map_w = (1 + (H + 2 * pad - HW) / stride)	# spactial width of conv result of one img by one filter.
	activation_map_h = (1 + (H + 2 * pad - HH) / stride) # spactial height of conv result of one img by one filter.

	streched_blk_size = C*HH*HW  # same as size of each filter neuron
	blk_num = activation_map_h * activation_map_w		 # streched blk number of one img.

	#	activation maps.shape (N, F, H', W')
	v = np.zeros((N,F,activation_map_h,activation_map_w))

	#	naive loop.

	for n in xrange(N): # loop the data batch

		for f in xrange(F):	# loop the filter number

			for a_h in np.arange(start=0, stop = activation_map_h): 		# loop vertically over the img

				for a_w in np.arange(start=0, stop= activation_map_w): 	#	slide horizontally over the img and conv
					v[n,f,a_h,a_w] = np.sum(x[n, :,\
								(a_h * stride):(a_h * stride + HH),	(a_w * stride):(a_w * stride + HW)] * w[f,:,:,:] ) + b[f]
					# i_h = a_h * stride
					# i_w = a_w * stride
					# x1 = x[n, :, (i_h):(i_h + HH),(i_w):(i_w + HW)]
					# w1 = w[f,:,:,:]

				#	end of horizontal slide

			#	end of vertical slide loop

	# reshape the result as out.
	# out = v.reshape()
	out = v
	# print '$$$$ conv_forward_naive -> out shape: ', out.shape
	# print '$$$ end of one fp $$$'
	# save cache

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	# NOTE: x include zero-padding, that's easy for bp calc !!!!
	cache = (x, w, b, conv_param)
	return out, cache


def conv_backward_naive(dout, cache):
	"""
	A naive implementation of the backward pass for a convolutional layer.

	Inputs:
	- dout: Upstream derivatives.
	- cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

	Returns a tuple of:
	- dx: Gradient with respect to x
	- dw: Gradient with respect to w
	- db: Gradient with respect to b
	"""
	dx, dw, db = None, None, None
	#############################################################################
	# TODO: Implement the convolutional backward pass.                          #
	#############################################################################
	pass
	#	kenneth yu

	#	recover cache
	# NOTE: x include zero-padding, that's easy for bp calc !!!!
	(x, w, b, conv_param) = cache
	# print('--- cached w:', w)

	# dout shape: (N,F,activation_map_h, activation_map_w)
	(N, F, activation_map_h, activation_map_w) = dout.shape
	(F,C,HH,HW) = w.shape
	(N,C,H,W) = x.shape			# NOTE: x include zero-padding, that's easy for bp calc !!!!

	#	init
	dw = np.zeros_like(w)
	dx = np.zeros_like(x)	# NOTE: x include zero-padding, that's easy for bp calc !!!!
	stride = conv_param.get('stride', 1)
	pad = conv_param.get('pad',0)

	# sanity check
	# print '==> conv bp dout shape: ', dout.shape
	# print '==> conv bp x shape: ', x.shape
	# print '==> conv bp w shape: ', w.shape
	# print '==> conv bp b shape: ', b.shape
	# print '==> conv bp stride: ', stride
	# print '==> conv bp pad: ', pad

	# calc dx, dw, db
	db = np.sum(dout, axis=(0,2,3))	# shape  (F,)

	for n in xrange(N):
		for f in xrange(F):
			for a_h in xrange(activation_map_h):
				for a_w in xrange(activation_map_w):
					# a_h, a_w is the spatial position of each activation map.
					do = dout[n,f,a_h,a_w]

					for c in xrange(C):
						for f_h in xrange(HH):
							for f_w in xrange(HW):
								#	f_h, f_w is the spatial  position of each filter.
								#	Note: conv is a merge of data at different spatial positions. so gradients
								#	should be	summed together.
								dx[n, c, a_h*stride + f_h, a_w*stride + f_w ] += do * w[f, c, f_h, f_w]
								dw[f, c, f_h, f_w ] +=  do * x[n, c, a_h*stride + f_h, a_w*stride + f_w]

							# end of f_w loop
						#end of f_h loop
					# end of c loop
				# end of a_w loop
			#end of a_h loop
		#end of f loop
	#end of n loop

	# sanity check:
	# print '==> conv bp dx shape before un-padding:', dx.shape		# NOTE: x include zero-padding, that's easy for bp calc !!!!
	# print '==> conv bp dx before un-padding:'	,dx
	# print '==> conv bp dw shape:', dw.shape
	# print '==> conv bp db shape:', db.shape

	# remove zero-padding of dx
	#	dx shape (N,C,H,W). just try delete.
	dx = np.delete(dx,slice(W-1,W-1-pad, -1),axis=-1)
	dx = np.delete(dx,slice(0, pad, 1),axis=-1)
	dx = np.delete(dx,slice(H-1, H-1-pad,-1),axis=-2)
	dx = np.delete(dx,slice(0, pad, 1),axis=-2)
	# print '==> conv bp dx after un-padding shape:', dx.shape		#
	# print dx

	#	save result
	# print '@@ end of one BP }'

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return dx, dw, db


def max_pool_forward_naive(x, pool_param):
	"""
	A naive implementation of the forward pass for a max pooling layer.

	Inputs:
	- x: Input data, of shape (N, C, H, W)
	- pool_param: dictionary with the following keys:
		- 'pool_height': The height of each pooling region
		- 'pool_width': The width of each pooling region
		- 'stride': The distance between adjacent pooling regions

	Returns a tuple of:
	- out: Output data
	- cache: (x, pool_param)
	"""
	out = None
	#############################################################################
	# TODO: Implement the max pooling forward pass                              #
	#############################################################################
	pass
	#	kenneth yu
	(N,C,H,W) = x.shape
	pool_h = pool_param.get('pool_height',0)
	pool_w = pool_param.get('pool_width',0)
	stride = pool_param.get('stride',1)

	#	calc the maxed output spatial size
	out_h = 1 + (H - pool_h)/stride
	out_w =	1 + (W - pool_w)/stride

	out_shape = (N,C,out_h, out_w)

	#	record the max out index. easy for bp.
	# out_idx = np.zeros_like(out)	# shape (N,C,out_h, out_w)
	out = np.zeros(out_shape)
	out_idx = [] # save idx for each max output. idx is tuple in the x.
	pool=[]
	max_idx_in_pool = 0

	# loop all the axises of x
	for n in xrange(N):
		for c in xrange(C):
			for o_h in xrange(out_h):	#	loop the output image height
				for o_w in xrange(out_w):	#	loop the output image width
					#	slice the pool from x. pool shape: (pool_h, pool_w)
					pool = x[n, c, o_h*stride:o_h*stride + pool_h, o_w*stride: o_w*stride + pool_w]
					out[n, c, o_h, o_w] = np.max(pool,axis=None)
					max_idx_in_pool = np.unravel_index(np.argmax(pool), (pool_h,pool_w))	#	tuple of index in pool.
					# max_idx_in_pool = np.where(pool==out[n, c, o_h, o_w])

					#	record the max index tuple in 'x'. when BP, need unravel.
					out_idx.append((n, c, o_h*stride+max_idx_in_pool[0],o_w*stride + max_idx_in_pool[1]))
					# print '@@ max pool idx:', max_idx_in_pool
					# print '@@ max value in pool: ',out[n, c, o_h, o_w]
					# print '@@ max value idx in x:',out_idx[-1]

	#	max from the pool
	# print ' >>> max out shape:', out.shape
	# print '	>>> max out idx in x len:', len(out_idx)

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	#	cache the index of max output
	# cache = (x, pool_param)
	cache = (x, out_idx,pool_param)

	return out, cache


def max_pool_backward_naive(dout, cache):
	"""
	A naive implementation of the backward pass for a max pooling layer.

	Inputs:
	- dout: Upstream derivatives
	- cache: A tuple of (x, pool_param) as in the forward pass.

	Returns:
	- dx: Gradient with respect to x
	"""
	dx = None
	#############################################################################
	# TODO: Implement the max pooling backward pass                             #
	#############################################################################
	pass
	#	kenneth yu
	#	unpack from cache
	(x, out_idx,pool_param) = cache
	(N,C,out_h, out_w) = dout.shape
	assert len(out_idx) == np.prod(dout.shape)

	#	sanity check
	# print '==> maxout bp dout shape:', dout.shape
	# print '==> maxout bp cached out_idx len:', len(out_idx)
	# print '==> maxout bp cached x shape:', x.shape
	#
	# print '==> maxout bp dout:', dout

	#	init
	dx = np.zeros_like(x)
	#	calc route switch. no params for maxout.only dx.
	# keep the same loop sequence n->c->o_h->o_w as the fp.
	for n in xrange(N):
		for c in xrange(C):
			for o_h in xrange(out_h):	#	loop the output image height
				for o_w in xrange(out_w):	#	loop the output image width
					idx = out_idx.pop(0)	# index tuple index in x corresponding to this max out.
					dx[idx] = dout[n, c, o_h, o_w]

	#	save data .
	# print '==> maxout bp dx:', dx
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
	"""
	Computes the forward pass for spatial batch normalization.

	Inputs:
	- x: Input data of shape (N, C, H, W)
	- gamma: Scale parameter, of shape (C,)
	- beta: Shift parameter, of shape (C,)
	- bn_param: Dictionary with the following keys:
		- mode: 'train' or 'test'; required
		- eps: Constant for numeric stability
		- momentum: Constant for running mean / variance. momentum=0 means that
			old information is discarded completely at every time step, while
			momentum=1 means that new information is never incorporated. The
			default of momentum=0.9 should work well in most situations.
		- running_mean: Array of shape (D,) giving running mean of features
		- running_var Array of shape (D,) giving running variance of features

	Returns a tuple of:
	- out: Output data, of shape (N, C, H, W)
	- cache: Values needed for the backward pass
	"""
	out, cache = None, None

	#############################################################################
	# TODO: Implement the forward pass for spatial batch normalization.         #
	#                                                                           #
	# HINT: You can implement spatial batch normalization using the vanilla     #
	# version of batch normalization defined above. Your implementation should  #
	# be very short; ours is less than five lines.                              #
	#############################################################################
	pass
	#	kenneth yu
	# see x as 'C' features , normalize over 'N * H * W' datas.
	#	api of vanilla bn above:
	# Input:
	# - x: Data of shape (N, D)
	# - gamma: Scale parameter of shape (D,)
	# - beta: Shift paremeter of shape (D,)
	# Returns a tuple of:
	# - out: of shape (N, D)
	# - cache: A tuple of values needed in the backward pass
	# we can see 'C' as 'D'
	(N, C, H, W) = x.shape
	# put C as the last axis to gurantee the reshape correctly.
	vanilla_out, cache = batchnorm_forward(x.transpose((0,2,3,1)).reshape((-1, C)), gamma, beta,bn_param)

	# print ' ==> bn spatial, input x shape :', x.shape
	# print ' ==> bn spatial, vanilla out shape :', vanilla_out.shape

	# print ' @@ original x: ',x
	# print ' @@  reshaped: x:', x.transpose((0,2,3,1)).reshape((-1, C))

	# this function api:
	# - out: Output data, of shape (N, C, H, W)
	#	vanilla_out shape: (N*H*M, C).
	out = vanilla_out.reshape((N,H,W,C)).transpose((0,3,1,2))
	# print ' ==> bn spatial, out shape :', out.shape

	# print ' @@ original vanilla_out: ', vanilla_out
	# print ' @@  reshaped: out:', out

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return out, cache


def spatial_batchnorm_backward(dout, cache):
	"""
	Computes the backward pass for spatial batch normalization.

	Inputs:
	- dout: Upstream derivatives, of shape (N, C, H, W)
	- cache: Values from the forward pass

	Returns a tuple of:
	- dx: Gradient with respect to inputs, of shape (N, C, H, W)
	- dgamma: Gradient with respect to scale parameter, of shape (C,)
	- dbeta: Gradient with respect to shift parameter, of shape (C,)
	"""
	dx, dgamma, dbeta = None, None, None

	#############################################################################
	# TODO: Implement the backward pass for spatial batch normalization.        #
	#                                                                           #
	# HINT: You can implement spatial batch normalization using the vanilla     #
	# version of batch normalization defined above. Your implementation should  #
	# be very short; ours is less than five lines.                              #
	#############################################################################
	pass
	#	kenneth yu
	#	vanilla bn bp api:
	# Inputs:
	# - dout: Upstream derivatives, of shape (N, D)
	# - cache: Variable of intermediates from batchnorm_forward.
	#
	# Returns a tuple of:
	# - dx: Gradient with respect to inputs x, of shape (N, D)
	# - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
	# - dbeta: Gradient with respect to shift parameter beta, of shape (D,)

	#	reshape dout.
	# we can see 'C' as 'D'.
	(N, C, H, W) = dout.shape
	dx, dgamma, dbeta =	batchnorm_backward_alt(dout.transpose((0,2,3,1)).reshape((-1,C)),cache)

	print ' -- > sp bn bp ,origin dout shape:', dout.shape
	print ' -- > sp bn bp ,dout reshape:', dout.transpose((0,2,3,1)).reshape((-1,C)).shape

	print ' -- > sp bn bp , vanilla dx shape:', dx.shape
	#	reshape dx
	dx = dx.reshape((N, H, W, C)).transpose((0,3,1,2))
	print ' -- > sp bn bp , final dx shape:', dx.shape ,' dgamma shape:', dgamma.shape, ' dbeta shape:', dbeta.shape

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return dx, dgamma, dbeta
  

def svm_loss(x, y):
	"""
	Computes the loss and gradient using for multiclass SVM classification.

	Inputs:
	- x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
		for the ith input.
	- y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
		0 <= y[i] < C

	Returns a tuple of:
	- loss: Scalar giving the loss
	- dx: Gradient of the loss with respect to x
	"""
	N = x.shape[0]
	correct_class_scores = x[np.arange(N), y]
	margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
	margins[np.arange(N), y] = 0
	loss = np.sum(margins) / N
	num_pos = np.sum(margins > 0, axis=1)
	dx = np.zeros_like(x)
	dx[margins > 0] = 1
	dx[np.arange(N), y] -= num_pos
	dx /= N     #dx is dscore
	return loss, dx


def softmax_loss(x, y):
	"""
	Computes the loss and gradient for softmax classification.

	Inputs:
	- x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
		for the ith input.
	- y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
		0 <= y[i] < C

	Returns a tuple of:
	- loss: Scalar giving the loss
	- dx: Gradient of the loss with respect to x
	"""
	#shift the data by max score of each data.

	probs = np.exp(x - np.max(x, axis=1, keepdims=True))

	# shifted_scores = (x - np.max(x, axis=1, keepdims=True))

	# kenneth yu . normalize to avoid inf problem.
	# scores_std = np.std(shifted_scores,axis= 1, keepdims=True)
	# scores_std +=  (1e-9) # avoid divide by zero
	# normalized_scores = shifted_scores / scores_std
	#
	# exp_probs = np.exp(normalized_scores)

	probs /= np.sum(probs, axis=1, keepdims=True)

	# print ' --- probs:' ,probs

	# probs = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)
	N = x.shape[0]
	# kenneth yu . how about the probs is close to zero then log will get inf? carefully weight
	# init and previous layers' BN will avoid large score differences?
	loss = -np.sum(np.log(probs[np.arange(N), y])) / N
	# xxx = np.log(probs[np.arange(N), y])
	# sss = -np.sum(xxx)
	# loss = sss / N

	dx = probs.copy()
	dx[np.arange(N), y] -= 1
	dx /= N

	# dx /= scores_std
	return loss, dx
