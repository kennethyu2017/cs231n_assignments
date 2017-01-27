import numpy as np

from cs231n import optim


class Solver(object):
	"""
	A Solver encapsulates all the logic necessary for training classification
	models. The Solver performs stochastic gradient descent using different
	update rules defined in optim.py.

	The solver accepts both training and validataion data and labels so it can
	periodically check classification accuracy on both training and validation
	data to watch out for overfitting.

	To train a model, you will first construct a Solver instance, passing the
	model, dataset, and various optoins (learning rate, batch size, etc) to the
	constructor. You will then call the train() method to run the optimization
	procedure and train the model.

	After the train() method returns, model.params will contain the parameters
	that performed best on the validation set over the course of training.
	In addition, the instance variable solver.loss_history will contain a list
	of all losses encountered during training and the instance variables
	solver.train_acc_history and solver.val_acc_history will be lists containing
	the accuracies of the model on the training and validation set at each epoch.

	Example usage might look something like this:

	data = {
		'X_train': # training data
		'y_train': # training labels
		'X_val': # validation data
		'X_train': # validation labels
	}
	model = MyAwesomeModel(hidden_size=100, reg=10)
	solver = Solver(model, data,
									update_rule='sgd',
									optim_config={
										'learning_rate': 1e-3,
									},
									lr_decay=0.95,
									num_epochs=10, batch_size=100,
									print_every=100)
	solver.train()


	A Solver works on a model object that must conform to the following API:

	- model.params must be a dictionary mapping string parameter names to numpy
		arrays containing parameter values.

	- model.loss(X, y) must be a function that computes training-time loss and
		gradients, and test-time classification scores, with the following inputs
		and outputs:

		Inputs:
		- X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
		- y: Array of labels, of shape (N,) giving labels for X where y[i] is the
			label for X[i].

		Returns:
		If y is None, run a test-time forward pass and return:
		- scores: Array of shape (N, C) giving classification scores for X where
			scores[i, c] gives the score of class c for X[i].

		If y is not None, run a training time forward and backward pass and return
		a tuple of:
		- loss: Scalar giving the loss
		- grads: Dictionary with the same keys as self.params mapping parameter
			names to gradients of the loss with respect to those parameters.
	"""

	def __init__(self, model, data, **kwargs):
		"""
		Construct a new Solver instance.

		Required arguments:
		- model: A model object conforming to the API described above
		- data: A dictionary of training and validation data with the following:
			'X_train': Array of shape (N_train, d_1, ..., d_k) giving training images
			'X_val': Array of shape (N_val, d_1, ..., d_k) giving validation images
			'y_train': Array of shape (N_train,) giving labels for training images
			'y_val': Array of shape (N_val,) giving labels for validation images

		Optional arguments:
		- update_rule: A string giving the name of an update rule in optim.py.
			Default is 'sgd'.
		- optim_config: A dictionary containing hyperparameters that will be
			passed to the chosen update rule. Each update rule requires different
			hyperparameters (see optim.py) but all update rules require a
			'learning_rate' parameter so that should always be present.
		- lr_decay: A scalar for learning rate decay; after each epoch the learning
			rate is multiplied by this value.
		- batch_size: Size of minibatches used to compute loss and gradient during
			training.
		- num_epochs: The number of epochs to run for during training.
		- print_every: Integer; training losses will be printed every print_every
			iterations.
		- verbose: Boolean; if set to false then no output will be printed during
			training.
		"""
		self.model = model
		self.X_train = data['X_train']
		self.y_train = data['y_train']
		self.X_val = data['X_val']
		self.y_val = data['y_val']

		# Unpack keyword arguments
		self.update_rule = kwargs.pop('update_rule', 'sgd')   # sgd as default.
		self.optim_config = kwargs.pop('optim_config', {})
		self.lr_decay = kwargs.pop('lr_decay', 1.0)
		self.batch_size = kwargs.pop('batch_size', 100)
		self.num_epochs = kwargs.pop('num_epochs', 10)

		self.print_every = kwargs.pop('print_every', 10)
		self.verbose = kwargs.pop('verbose', True)

		# Throw an error if there are extra keyword arguments
		if len(kwargs) > 0:
			extra = ', '.join('"%s"' % k for k in kwargs.keys())
			raise ValueError('Unrecognized arguments %s' % extra)

		# Make sure the update rule exists, then replace the string
		# name with the actual function
		if not hasattr(optim, self.update_rule):
			raise ValueError('Invalid update_rule "%s"' % self.update_rule)
		# replace the string name with the actual function.
		self.update_rule = getattr(optim, self.update_rule)

		self._reset()


	def _reset(self):
		"""
		Set up some book-keeping variables for optimization. Don't call this
		manually.
		"""
		# Set up some variables for book-keeping
		self.epoch = 0
		self.best_val_acc = 0
		self.best_params = {}
		self.loss_history = []
		self.train_acc_history = []
		self.val_acc_history = []
		#	kenneth yu
		self.layer_history = {}
		# self.layer_b_history = []
		# self.layer_dw_history = []
		# self.layer_db_history = []
		# self.m_history = []
		# self.v_history = []
		# randomly choose the W and b specific element.

		self.plot_layer = 1 if (len(self.model.params)/4)  == 0 \
													else np.random.randint(1, (len(self.model.params)/4) + 1)
		self.plot_w_name = 'W' + str(self.plot_layer)
		self.plot_b_name = 'b' + str(self.plot_layer)
		self.plot_dw_name = 'dw' + str(self.plot_layer)
		self.plot_db_name = 'db' + str(self.plot_layer)
		self.plot_w_m_name = 'W' + str(self.plot_layer) + 'm'
		self.plot_w_v_name = 'W' +str(self.plot_layer) + 'v'
		self.plot_b_m_name = 'b' + str(self.plot_layer) + 'm'
		self.plot_b_v_name = 'b' +str(self.plot_layer) + 'v'

		self.layer_history[self.plot_w_name] = []

		self.layer_history[self.plot_b_name] = []
		self.layer_history[self.plot_dw_name] = []
		self.layer_history[self.plot_db_name] = []
		self.layer_history[self.plot_w_m_name ]= []
		self.layer_history[self.plot_w_v_name] = []
		self.layer_history[self.plot_b_m_name ]= []
		self.layer_history[self.plot_b_v_name ]= []

		self.plot_w_idx = (np.random.randint(0, self.model.params[self.plot_w_name].shape[0] ),
											 np.random.randint(0, self.model.params[self.plot_w_name].shape[1] ))
		self.plot_b_idx = np.random.randint(0, self.model.params[self.plot_b_name].shape[0])

		# print '\n ### layer plot position: layer: %d ' % self.plot_layer, 'W-index:', self.plot_w_idx, 'b-index: ',self.plot_b_idx


		# Make a deep copy of the optim_config for each parameter
		self.optim_configs = {}
		#	WHY not use self.copy()
		for p in self.model.params:
			# generate d ;optim_config is a dict containing initial hyper-parameter for each layer parameter,
			# required by optim.
			#	same init config for all the layers.
			d = {k:v for k, v in self.optim_config.iteritems()}
			# save optim_config hyper-parameters to each layer-params into optim_configs.
			# each layer will have some hyper-parameter of its own, such as RMS_prop cache, SGD momentum velocity,...
			self.optim_configs[p] = d


	def _step(self,layer_verbose=False, layer_plot=False):
		"""
		Make a single gradient update. This is called by train() and should not
		be called manually.
		"""
		# Make a minibatch of training data
		num_train = self.X_train.shape[0]
		# we need the same random mask for both X and y, so dont choice on num_train directly.
		batch_mask = np.random.choice(num_train, self.batch_size)
		X_batch = self.X_train[batch_mask]
		y_batch = self.y_train[batch_mask]

		# Compute loss and gradient
		loss, grads = self.model.loss(X_batch, y_batch, layer_verbose)
		self.loss_history.append(loss)



		# print 'Training == check grad W2 [28,3] :' , grads['W2'][28,3]
		# print 'Training == check grad W3 [38,7] :' , grads['W2'][38,7]
		# print 'Training == check grad W5 [48,4] :' , grads['W2'][48,4]


		# Perform a parameter update
		# params is a dictionary, so (p, w) is a key-value pair.
		for p, w in self.model.params.iteritems():
			#	kenneth yu. Last output layer not have BN so ignore gamma, beta.
			# if grads.has_key[p]:
			dw = grads[p]
			# each model.params is a layer-param-groups, e.g. fc-layer weights and bias. we save hyper-params for each
			# layer-param-groups in optim_configs, index by param name.
			config = self.optim_configs[p]
			# update_rule defined in optim.py

			next_w, next_config = self.update_rule(w, dw, config)

			self.model.params[p] = next_w
			self.optim_configs[p] = next_config

		#	kenneth yu. plot tracking a specific parameter.
		if layer_plot:
			w = self.model.params[self.plot_w_name]
			self.layer_history[self.plot_w_name].append(w[self.plot_w_idx])

			b = self.model.params[self.plot_b_name]
			self.layer_history[self.plot_b_name].append(b[self.plot_b_idx])

			dw = grads[self.plot_w_name]
			self.layer_history[self.plot_dw_name].append(dw[self.plot_w_idx])

			db = grads[self.plot_b_name]
			self.layer_history[self.plot_db_name].append(db[self.plot_b_idx])

			if self.update_rule.func_name == 'adam':
				w_configs = self.optim_configs[self.plot_w_name]
				self.layer_history[self.plot_w_m_name].append(w_configs['m'][self.plot_w_idx] )
				self.layer_history[self.plot_w_v_name].append(w_configs['v'][self.plot_w_idx])

				b_config = self.optim_configs[self.plot_b_name]
				self.layer_history[self.plot_b_m_name].append(b_config['m'][self.plot_b_idx])
				self.layer_history[self.plot_b_v_name].append(b_config['v'][self.plot_b_idx])

		##	end of layer plot.


	def check_accuracy(self, X, y, num_samples=None, batch_size=100):
		"""
		Check accuracy of the model on the provided data.

		Inputs:
		- X: Array of data, of shape (N, d_1, ..., d_k)
		- y: Array of labels, of shape (N,)
		- num_samples: If not None, subsample the data and only test the model
			on num_samples datapoints.
		- batch_size: Split X and y into batches of this size to avoid using too
			much memory.

		Returns:
		- acc: Scalar giving the fraction of instances that were correctly
			classified by the model.
		"""

		# Maybe subsample the data
		N = X.shape[0]
		if num_samples is not None and N > num_samples:
			mask = np.random.choice(N, num_samples)
			N = num_samples
			X = X[mask]		#shape (num_samples, d_1, ..., d_k)
			y = y[mask]		#shape (num_samples, )

		# Compute predictions in batches
		num_batches = N / batch_size
		if N % batch_size != 0:
			num_batches += 1
		y_pred = []
		for i in xrange(num_batches):
			start = i * batch_size
			end = (i + 1) * batch_size
			scores = self.model.loss(X[start:end])
			y_pred.append(np.argmax(scores, axis=1))		# y_pred: a list of arrays.
		y_pred = np.hstack(y_pred)					# concatenate along second-axis.
		acc = np.mean(y_pred == y)

		return acc


	def train(self, layer_verbose=False,layer_plot=False):
		"""
		Run optimization to train the model.
		"""
		num_train = self.X_train.shape[0]
		iterations_per_epoch = max(num_train / self.batch_size, 1)
		# one epoch means batch-data training iterates the whole training data set.
		num_iterations = self.num_epochs * iterations_per_epoch

		for t in xrange(num_iterations):
			# we train through a mini batch on each step.
			self._step(layer_verbose, layer_plot)

			# Maybe print training loss
			if self.verbose and t % self.print_every == 0:
				print '(Training ===  Iteration %d / %d) loss: %f' % (
							 t + 1, num_iterations, self.loss_history[-1])		#[-1] meas last element.

			# At the end of every epoch, increment the epoch counter and decay the
			# learning rate.
			#	Kenneth yu : WHY NOT half the l_r every 5-epochs???
			epoch_end = (t + 1) % iterations_per_epoch == 0
			if epoch_end:
				self.epoch += 1
				for k in self.optim_configs:
					##	l_r is per paramter hyper-parameter. so we update l_r for each layer parameter group.
					# old_l_r = self.optim_configs[k]['learning_rate']
					self.optim_configs[k]['learning_rate'] *= self.lr_decay
					# print 'Training === decay learning rate: from %e to %e' % (old_l_r, self.optim_configs[k]['learning_rate'])
				# print 'Training === check W2[28,3] : ',  self.model.params['W2'][28,3]
				# print 'Training === check W3[38,7] : ',  self.model.params['W2'][38,7]
				# print 'Training === check W5[48,4] : ',  self.model.params['W2'][48,4]

			# Check train and val accuracy on the first iteration, the last
			# iteration, and at the end of each epoch.
			first_it = (t == 0)
			last_it = (t == num_iterations + 1)
			if first_it or last_it or epoch_end:
				# use 1000 training samples to check accu
				train_acc = self.check_accuracy(self.X_train, self.y_train,
																				num_samples=1000)
				# use whole val samples to check accu
				val_acc = self.check_accuracy(self.X_val, self.y_val)
				self.train_acc_history.append(train_acc)
				self.val_acc_history.append(val_acc)

				# if self.verbose:
				# 	print 'Training === (Epoch %d / %d) train acc: %f; val_acc: %f' % (
				# 				 self.epoch, self.num_epochs, train_acc, val_acc)

				# Keep track of the best model. by comparing the val acc:
				if val_acc > self.best_val_acc:
					self.best_val_acc = val_acc
					self.best_params = {}
					for k, v in self.model.params.iteritems():
						#copy all the params. k : the param name,e.g.W1,b1, W2, b2... v: corresponding value.
						self.best_params[k] = v.copy()
						####		kenneth yu : need record the hyper-p?

		# At the end of training swap the best params into the model
		self.model.params = self.best_params

		#	kenneth yu
		import matplotlib.pyplot as plt
		if layer_plot:
			plt.figure(figsize=(24,13))
			total_plots = len(self.layer_history)
			i = 1
			for (k,v) in sorted(self.layer_history.iteritems()):
				if k == self.plot_w_name:
					k += str(self.plot_w_idx)
				if k == self.plot_b_name:
					k += '(' + str(self.plot_b_idx)  + ')'

				plt.subplot(2, np.ceil(total_plots/2.0),i)
				i += 1
				plt.title(k)
				plt.plot(v, 'r-')
				# plt.legend(loc='lower left')

			plt.show()


