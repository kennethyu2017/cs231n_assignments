import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import cs231n.data_utils as data
from cs231n.classifiers.pretrained_cnn import PretrainedCNN
from cs231n.image_utils import blur_image,deprocess_image, preprocess_image

# Image Generation
# In this notebook we will continue our exploration of image gradients using the deep model that was pretrained on
# TinyImageNet. We will explore various ways of using these image gradients to generate images.
#  We will implement class visualizations, feature inversion, and DeepDream.
# TinyImageNet and pretrained model. As in the previous notebook, load the TinyImageNet dataset and the pretrained
# model.

# data need 2.8GB memory!
model = PretrainedCNN(h5_file='cs231n/datasets/pretrained_model.h5')
data = data.load_tiny_imagenet('cs231n/datasets/tiny-imagenet-100-A', subtract_mean=True,num_object_tr_images=5)

# Class activation visualization
# By starting with a random noise image and performing gradient ascent on a target class, we can generate an image
#  that the network will recognize as the target class. This idea was first presented in [1]; [2] extended this idea by
# suggesting several regularization techniques that can improve the quality of the generated image.

# Concretely, let $I$ be an image and let y be a target class. Let $s_y(I)$ be the score that a convolutional
# network assigns to the image I for class y; note that these are raw unnormalized scores, not class probabilities.
# (according to [1] , by using the un-normalized scores we can achieve better result. and class prob can be achieved
# larger by decrease the scores of irrevelant classes.)
#  We wish to generate an image  I that achieves a high score for the class y by solving the problem
# I = argmax_I (score_y(I) + R(I))

# where R is a (possibly implicit) regularizer. We can solve this optimization problem using gradient descent,
# computing gradients with respect to the generated image. We will use (explicit) L2 regularization of the form
# R(I) = lambda * ||I||^2
#
# and implicit regularization as suggested by [2] by periodically blurring the generated image. We can solve this
# problem using gradient ascent on the generated image.
#
# In the cell below, complete the implementation of the `create_class_visualization` function.
#
# [1] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. "Deep Inside Convolutional Networks: Visualising
# Image Classification Models and Saliency Maps", ICLR Workshop 2014.
#
# [2] Yosinski et al, "Understanding Neural Networks Through Deep Visualization", ICML 2015 Deep Learning Workshop


def create_class_visualization(target_y, model, **kwargs):
	#
  # Perform optimization over the image to generate class visualizations.
	#
  # Inputs:
  # - target_y: Integer in the range [0, 100) giving the target class
  # - model: A PretrainedCNN that will be used for generation
	#
  # Keyword arguments:
  # - learning_rate: Floating point number giving the learning rate
  # - blur_every: An integer; how often to blur the image as a regularizer
  # - l2_reg: Floating point number giving L2 regularization strength on the image;
  #   this is lambda in the equation above.
  # - max_jitter: How much random jitter to add to the image as regularization
  # - num_iterations: How many iterations to run for
  # - show_every: How often to show the image

  learning_rate = kwargs.pop('learning_rate', 10000)
  blur_every = kwargs.pop('blur_every', 1)
  l2_reg = kwargs.pop('l2_reg', 1e-6)
  max_jitter = kwargs.pop('max_jitter', 4)
  num_iterations = kwargs.pop('num_iterations', 100)
  show_every = kwargs.pop('show_every', 25)

  print '===data[mean_image]:', data['mean_image']

  # init X to standard norm . keep zero-mean as training data.
  X = np.random.randn(1, 3, 64, 64)
  for t in xrange(num_iterations):
    # As a regularizer, add random jitter to the image
    # should be a kind of data pre-processing to introduce more noise?
    [ox, oy] = np.random.randint(-max_jitter, max_jitter+1, 2)
    # jittering: randomly roll the pixels: roll ox pixels along Width ,roll oy along Height.
    X = np.roll(np.roll(X, ox, -1), oy, -2)   # X shape (1,3,64,64)

    dX = None
    ############################################################################
    # TODO: Compute the image gradient dX of the image with respect to the     #
    # target_y class score. This should be similar to the fooling images. Also #
    # add L2 regularization to dX and update the image X using the image       #
    # gradient and the learning rate.                                          #
    ############################################################################
    pass
    # kenneth yu
    # optimize object: argmax( activation_of_unit_i(I) + R(I) ). for output layer , the activation is the
    # class score of i-classfier-unit.
    # 1.init image X.already done above
    # and introduce noise as regu: randomly jitter generative image. done above.

    # 2. Feedforward and get scores of all classes.
    # because we fixed the model parameters, so use the mode as test.(will affect the BN)
    # scores shape: (1, C)
    scores, cache = model.forward(X, mode='test')  # output of last affine layer is scores of each class.

    # print 'shape of scores:', scores.shape
    if t%show_every == 0:
      print 'score of class %d : %f' % (target_y, scores[0,target_y])

    # 3. calc d_score_c and BP to get dX.
    # we only count the d of the target class.
    dscores = np.zeros_like(scores) # same shape as scores (1,100)
    dscores[0,target_y] = 1
    dX, _ = model.backward(dscores, cache)

    # 4. calc grads with reg:
    grads_X = dX - 2 * l2_reg * X

    # 5. grads ascend with l_r
    X += learning_rate * grads_X

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # Undo the jitter, why ???
    X = np.roll(np.roll(X, -ox, -1), -oy, -2)

    # kenneth yu:Incorrect??? As a regularizer. clip interval: [-mean, 255-mean]
    # TODO:check yosinski's code.
    if t % show_every ==0:
      old_X = X
      X = np.clip(X, -data['mean_image'], 255.0 - data['mean_image'])
      print 'X clipped elements #:', np.sum(X != old_X)

    # As a regularizer, periodically blur the image
    if t % blur_every == 0:
      X = blur_image(X)

    # Periodically show the image
    if t % show_every == 0:
      plt.imshow(deprocess_image(X, data['mean_image']))
      plt.gcf().set_size_inches(3, 3)
      plt.axis('off')
      plt.show()
    #end of iteration
  return X

# You can use the code above to generate some cool images! An example is shown below. Try to generate a cool-looking
# image. If you want you can try to implement the other regularization schemes from Yosinski et al, but it isn't required.

# target_y = 43 # Tarantula
target_y = 67 #try .
print data['class_names'][target_y]
X = create_class_visualization(target_y, model, show_every=25)

print '=== the reulst X:' , X
plt.imshow(deprocess_image(X, data['mean_image']))
plt.axis('off')
plt.title('The class : %d' % (target_y))
plt.show()


# Feature Inversion
# In an attempt to understand the types of features that convolutional networks learn to recognize, a recent
#  paper [1] attempts to reconstruct an image from its feature representation. We can easily implement this idea using
#  image gradients from the pretrained network.
#
# Concretely, given a image I, let ph_i(I) be the activations at layer i of the convolutional network.
# We wish to find an image I* with a similar feature representation as I at layer i of the
# network by solving the optimization problem
# I* = argmin_{I'} (  Eucliden_Norm(ph_i(I) - ph_i(I'))^2 + R(I'))
#
# where is the squared Euclidean norm. As above, R is a (possibly implicit) regularizer. We can solve
# this optimization problem using gradient descent, computing gradients with respect to the generated image. We will use
# (explicit) L2 regularization of the form (no Total Variation regu in [2] )
# R(I') += lambda * |I'|_2^2
#
# together with implicit regularization by periodically blurring the image, as recommended by [2].
#
# Implement this method in the function below.
#
# [1] Aravindh Mahendran, Andrea Vedaldi, "Understanding Deep Image Representations by Inverting them", CVPR 2015
#
# [2] Yosinski et al, "Understanding Neural Networks Through Deep Visualization", ICML 2015 Deep Learning Workshop

def invert_features(target_feats, layer, model, **kwargs):
  """
  Perform feature inversion in the style of Mahendran and Vedaldi 2015, using
  L2 regularization and periodic blurring.

  Inputs:
  - target_feats: Image features of the target image, of shape (1, C, H, W);
    we will try to generate an image that matches these features
  - layer: The index of the layer from which the features were extracted
  - model: A PretrainedCNN that was used to extract features

  Keyword arguments:
  - learning_rate: The learning rate to use for gradient descent
  - num_iterations: The number of iterations to use for gradient descent
  - l2_reg: The strength of L2 regularization to use; this is lambda in the
    equation above.
  - blur_every: How often to blur the image as implicit regularization; set
    to 0 to disable blurring.
  - show_every: How often to show the generated image; set to 0 to disable
    showing intermediate reuslts.

  Returns:
  - X: Generated image of shape (1, 3, 64, 64) that matches the target features.
  """

  learning_rate = kwargs.pop('learning_rate', 10000)
  num_iterations = kwargs.pop('num_iterations', 500)
  l2_reg = kwargs.pop('l2_reg', 1e-7)
  blur_every = kwargs.pop('blur_every', 1)
  show_every = kwargs.pop('show_every', 50)

  #init to standard norm random. keep zero-mean as training data.
  X = np.random.randn(1, 3, 64, 64)
  for t in xrange(num_iterations):
    ############################################################################
    # TODO: Compute the image gradient dX of the reconstruction loss with      #
    # respect to the image. You should include L2 regularization penalizing    #
    # large pixel values in the generated image using the l2_reg parameter;    #
    # then update the generated image using the learning_rate from above.      #
    ############################################################################
    pass
    #kennthyu
    #TODO: why not do jittering?
    #pre-processing data. need normalize target_feats ?

    #initial data, params.init X to standard norm. Normalize to same scale.

    #FF. get activation.
    (feats, cache) = model.forward(X, start=None, end=layer, mode='test')

    #calc l2 loss: |feats-target_feats|^2
    loss = (np.linalg.norm(feats-target_feats)) ** 2
    if t % show_every == 0:
      print '--iteration: %d  loss: %e ' % (t, loss)
      print '+max of X', np.max(X), '--min of X', np.min(X)

    #calc direvative of loss
    d_loss_on_feats = 2 *(feats-target_feats) * 1

    #BP
    d_loss_on_X, _ = model.backward(d_loss_on_feats, cache)

    #add l2_regu item. re norm to be same scale ??
    #TODO : get rid of SPIKE.
    grads = d_loss_on_X + l2_reg * 2 * X

    #SGD to update image data.
    X -= learning_rate * grads
    if t % show_every == 0:
      print '++max of X', np.max(X), '--min of X', np.min(X)


    #post-process data: renorm, add mean,

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # As a regularizer, clip the image(X should subtract mean, so we clip it to [0,255] - mean )
    X = np.clip(X, -data['mean_image'], 255.0 - data['mean_image'])
    if t % show_every == 0:
      print '+++max of X', np.max(X), '--min of X', np.min(X)

    # According to [2] As a regularizer, periodically blur the image
    if (blur_every > 0) and t % blur_every == 0:
      X = blur_image(X)

    if (show_every > 0) and (t % show_every == 0 or t + 1 == num_iterations):
      print '++++max of X', np.max(X), '--min of X', np.min(X)
      print '++++mean of X', X.mean()
      plt.imshow(deprocess_image(X, data['mean_image']))
      plt.gcf().set_size_inches(3, 3)
      plt.axis('off')
      plt.title('t = %d' % t)
      plt.show()
  #end of iteration.


### Shallow feature reconstruction
# After implementing the feature inversion above, run the following cell to try and reconstruct features from the
# fourth convolutional layer of the pretrained model. You should be able to reconstruct  the features using the provided
# optimization parameters.



filename = 'kitten.jpg'
end_layer = 3 # layers start from 0 so these are features after 4 convolutions

img_obj = Image.open(filename,'r').resize((64, 64))

img = np.asarray(img_obj) # convert Image object to np array with shape (64,64,3) through the __array_interface__
# attribute api.
# img = np.array(getattr(img_obj, '__array_interface__'))

plt.imshow(img)
plt.gcf().set_size_inches(3, 3)
plt.title('Original image--kitten.jpg')
plt.axis('off')
plt.show()

# Preprocess the image before passing it to the network:
# subtract the mean, add a dimension, etc
img_pre = preprocess_image(img, data['mean_image'])

# Extract features from the image
feats, _ = model.forward(img_pre, end=end_layer)

# Invert the features
kwargs = {
  'num_iterations': 400,
  'learning_rate': 5000,
  'l2_reg': 1e-8,
  'show_every': 100,
  'blur_every': 10,
}
# unpack the kwargs dict.
X = invert_features(feats, end_layer, model, **kwargs)



### Deep feature reconstruction
# Reconstructing images using features from deeper layers of the network tends to give interesting results. In the cell
#  below, try to reconstruct the best image you can by inverting the features after 7 layers of convolutions. You will
#  need to play with the hyperparameters to try and get a good result.
#
# HINT: If you read the paper by Mahendran and Vedaldi, you'll see that reconstructions from deep features tend not to
# look much like the original image, so you shouldn't expect the results to look like the reconstruction above. You
# should be able to get an image that shows some discernable structure within 1000 iterations.

filename = 'kitten.jpg'
layer = 6 # layers start from 0 so these are features after 7 convolutions


img_obj = Image.open(filename,'r').resize((64, 64))

img = np.asarray(img_obj) # convert Image object to np array with shape (64,64,3) through the __array_interface__
# attribute api.
# img = np.array(getattr(img_obj, '__array_interface__'))

plt.imshow(img)
plt.gcf().set_size_inches(3, 3)
plt.title('Original image')
plt.axis('off')
plt.show()

# Preprocess the image before passing it to the network:
# subtract the mean, add a dimension, etc
img_pre = preprocess_image(img, data['mean_image'])

# Extract features from the image
feats, _ = model.forward(img_pre, end=layer)

# Invert the features
# You will need to play with these parameters.
kwargs = {
  'num_iterations': 1000,
  'learning_rate': 1e4,
  'l2_reg': 1e-8,
  'show_every': 100,
  'blur_every': 50,
}
X = invert_features(feats, layer, model,**kwargs)

# DeepDream
# In the summer of 2015, Google released a [blog post](http://googleresearch.blogspot.com/2015/06/
# inceptionism-going-deeper-into-neural.html) describing a new method of generating images from neural networks,
# and they later [released code](https://github.com/google/deepdream) to generate these images.
#
# The idea is very simple. We pick some layer from the network, pass the starting image through the network to extract
# features at the chosen layer, set the gradient at that layer equal to the activations themselves, and then
# backpropagate to the image. This has the effect of modifying the image to amplify the activations at the chosen
# layer of the network.
#
# For DeepDream we usually extract features from one of the convolutional layers, allowing us to generate images of
#  any resolution.
#
# We can implement this idea using our pretrained network. The results probably won't look as good as Google's since
#  their network is much bigger, but we should still be able to generate some interesting images.
#

def deepdream(X, layer, model, **kwargs):
  """
  Generate a DeepDream image.

  Inputs:
  - X: Starting image, of shape (1, 3, H, W)
  - layer: Index of layer at which to dream
  - model: A PretrainedCNN object

  Keyword arguments:
  - learning_rate: How much to update the image at each iteration
  - max_jitter: Maximum number of pixels for jitter regularization
  - num_iterations: How many iterations to run for
  - show_every: How often to show the generated image
  """

  X = X.copy()

  learning_rate = kwargs.pop('learning_rate', 5.0)
  max_jitter = kwargs.pop('max_jitter', 16)
  num_iterations = kwargs.pop('num_iterations', 100)
  show_every = kwargs.pop('show_every', 25)

  for t in xrange(num_iterations):
    # As a regularizer, add random jitter to the image
    ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
    X = np.roll(np.roll(X, ox, -1), oy, -2)

    dX = None
    ############################################################################
    # TODO: Compute the image gradient dX using the DeepDream method. You'll   #
    # need to use the forward and backward methods of the model object to      #
    # extract activations and set gradients for the chosen layer. After        #
    # computing the image gradient dX, you should use the learning rate to     #
    # update the image X.                                                      #
    ############################################################################
    pass
    #kenneth yu
    # idea:We pick some layer from the network, pass the starting image through the network to extract
    # features at the chosen layer, set the gradient at that layer equal to the activations themselves, and then
    # backpropagate to the image.

    # init the data. and preprocess , normalize....

    # FF. get feats(activation) of layer.
    feats, cache = model.forward(X, start=None,end=layer, mode='test')
    # if t % 10:
    #   print 'iteration :%d, feats^2:' % (t), feats ** 2

    # optimize objective:  argmax ( 1/2 * feats ^ 2 - R(X)).
    # BP get grads. dout of layer is just feats.
    # dX, _ = model.backward(feats, cache)
    dX,_ = model.backward(np.ones_like(feats),cache)

    #TODO: add reg_strength * regu item

    #gradient ascent to  update input by l_r * grads
    X += learning_rate * dX

    #end of iteration.

    #post process image. re-normalize. add mean. for visualiztion.

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # Undo the jitter
    X = np.roll(np.roll(X, -ox, -1), -oy, -2)

    # As a regularizer, clip the image
    mean_pixel = data['mean_image'].mean(axis=(1, 2), keepdims=True)
    X = np.clip(X, -mean_pixel, 255.0 - mean_pixel)

    # TODO: blur every xx iterations.

    # Periodically show the image
    if t == 0 or (t + 1) % show_every == 0:
      img = deprocess_image(X, data['mean_image'], mean='pixel')
      # plt.imshow(img)
      # plt.title('t = %d' % (t + 1))
      # plt.gcf().set_size_inches(6, 6)
      # plt.title('iteration: %d' % (t))
      # plt.axis('off')
      # plt.show()
      img_obj = Image.fromarray(img)
      img_obj.save('/Users/kennethyu/Desktop/mm_deepdream/mm_it%d.jpg' % (t))
  return X

# Generate some images!
# Try and generate a cool-looking DeepDeam image using the pretrained network. You can try using different layers, or
#  starting from different images. You can reduce the image size if it runs too slowly on your machine, or increase
# the image size if you are feeling ambitious.


def read_image(filename, max_size):
  """
  Read an image from disk and resize it so its larger side is max_size
  """
  img_obj = Image.open(filename,'r')
  W, H= img_obj.size
  if H >= W:
    img_obj = img_obj.resize((max_size, int(W * float(max_size) / H)))  # size (W,H)
  elif H < W:
    img_obj = img_obj.resize((int(H * float(max_size) / W), max_size))
  return np.asarray(img_obj)

#TODO: use your image to try
print 'pls specify your image file path and name....'
filename = '/Users/kennethyu/Desktop/mm1.jpg'
# filename = '~/Desktop/mm1.jpg'
max_size = 256
img = read_image(filename, max_size)
plt.title('Original image  '+'filename')
plt.imshow(img)
plt.axis('off')

# Preprocess the image by converting to float, transposing,
# and performing mean subtraction.
img_pre = preprocess_image(img, data['mean_image'], mean='pixel')
out = deepdream(img_pre, 7, model, num_iterations=1000,learning_rate=2000,show_every=50)








