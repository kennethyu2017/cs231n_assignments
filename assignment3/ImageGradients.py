# Image Gradients
# In this notebook we'll introduce the TinyImageNet dataset and a deep CNN that has been pretrained on this dataset.
#  You will use this pretrained model to compute gradients with respect to images, and use these image gradients
#  to produce class saliency maps and fooling images.

# As usual, a bit of setup

import time, os, json
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mtplt


from cs231n.classifiers.pretrained_cnn import PretrainedCNN
from cs231n.data_utils import load_tiny_imagenet
from cs231n.image_utils import blur_image, deprocess_image

# Introducing TinyImageNet
#
# The TinyImageNet dataset is a subset of the ILSVRC-2012 classification dataset. It consists of 200 object classes,
# and for each object class it provides 500 training images, 50 validation images, and 50 test images.(so totaly 1M
# training images, 100k val images, and 100k test images.)
#  All images have been downsampled to 64x64 pixels. We have provided the labels for all training and validation images,
#  but have withheld the labels for the test images.
#
# We have further split the full TinyImageNet dataset into two equal pieces, each with 100 object classes. We refer to
#  these datasets as TinyImageNet-100-A and TinyImageNet-100-B; for this exercise you will work with TinyImageNet-100-A.
#
# To download the data, go into the `cs231n/datasets` directory and run the script `get_tiny_imagenet_a.sh`. Then run
#  the following code to load the TinyImageNet-100-A dataset into memory.
#
# NOTE: The full TinyImageNet-100-A dataset will take up about 250MB of disk space, and loading the full
#  TinyImageNet-100-A dataset into memory will use about 2.8GB of memory.
#

# need 2.8GB memory
# kenneth yu. load small
# data = load_tiny_imagenet('cs231n/datasets/tiny-imagenet-100-A',subtract_mean=True)
data = load_tiny_imagenet('cs231n/datasets/tiny-imagenet-100-A', num_object_tr_images=5, subtract_mean=True)


# TinyImageNet-100-A classes
# Since ImageNet is based on the WordNet ontology, each class in ImageNet (and TinyImageNet) actually has several
# different names. For example "pop bottle" and "soda bottle" are both valid names for the same class. Run the
# following to see a list of all classes in TinyImageNet-100-A:

for i, names in enumerate(data['class_names']):
  print i, ' '.join('"%s"' % name for name in names)

# Visualize Examples
# Run the following to visualize some example images from random classses in TinyImageNet-100-A. It selects
#  classes and images randomly, so you can run it several times to see different images.

# Visualize some examples of the training data
classes_to_show = 7
examples_per_class = 5

class_idxs = np.random.choice(len(data['class_names']), size=classes_to_show, replace=False)
for i, class_idx in enumerate(class_idxs):
  train_idxs, = np.nonzero(data['y_train'] == class_idx)
  train_idxs = np.random.choice(train_idxs, size=examples_per_class, replace=False)
  for j, train_idx in enumerate(train_idxs):
    img = deprocess_image(data['X_train'][train_idx], data['mean_image'])
    plt.subplot(examples_per_class, classes_to_show, 1 + i + classes_to_show * j)
    if j == 0:
      plt.title(data['class_names'][class_idx][0])
    plt.imshow(img)
    plt.gca().axis('off')

plt.show()

# Pretrained model
# We have trained a deep CNN for you on the TinyImageNet-100-A dataset that we will use for image visualization.
#  The model has 9 convolutional layers (with spatial batch normalization) and 1 fully-connected hidden layer
#  (with batch normalization).

# To get the model, run the script `get_pretrained_model.sh` from the `cs231n/datasets` directory. After doing so,
# run the following to load the model from disk.

# kenneth yu:model is trained on 100-A , so scores have 100-classes.
model = PretrainedCNN(h5_file='cs231n/datasets/pretrained_model.h5')

## Pretrained model performance
# Run the following to test the performance of the pretrained model on some random training and validation set images.
#  You should see training accuracy around 90% and validation accuracy around 60%; this indicates a bit of overfitting,
#  but it should work for our visualization experiments.

batch_size = 100

# Test the model on training data
mask = np.random.randint(data['X_train'].shape[0], size=batch_size)
X, y = data['X_train'][mask], data['y_train'][mask]
y_pred = model.loss(X).argmax(axis=1)
print 'Training accuracy: ', (y_pred == y).mean()

# Test the model on validation data
mask = np.random.randint(data['X_val'].shape[0], size=batch_size)
X, y = data['X_val'][mask], data['y_val'][mask]
y_pred = model.loss(X).argmax(axis=1)
print 'Validation accuracy: ', (y_pred == y).mean()


# Saliency Maps
# Using this pretrained model, we will compute class saliency maps as described in Section 3.1 of [1].
#
# As mentioned in Section 2 of the paper, you should compute the gradient of the image with respect to the
# unnormalized class score, not with respect to the normalized class probability.
#
# You will need to use the `forward` and `backward` methods of the `PretrainedCNN` class to compute gradients with
# respect to the image. Open the file `cs231n/classifiers/pretrained_cnn.py` and read the documentation for these
# methods to make sure you know how they work. For example usage, you can see the `loss` method. Make sure to run the
#  model in `test` mode when computing saliency maps.
#
# [1] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. "Deep Inside Convolutional Networks: Visualising
# Image Classification Models and Saliency Maps", ICLR Workshop 2014.
def compute_saliency_maps(X, y, model):

  # Compute a class saliency map using the model for images X and labels y.
	#
  # Input:
  # - X: Input images, of shape (N, 3, H, W)
  # - y: Labels for X, of shape (N,)
  # - model: A PretrainedCNN that will be used to compute the saliency map.
	#
  # Returns:
  # - saliency: An array of shape (N, H, W) giving the saliency maps for the input
  #   images.

  saliency = None
  ##############################################################################
  # TODO: Implement this function. You should use the forward and backward     #
  # methods of the PretrainedCNN class, and compute gradients with respect to  #
  # the unnormalized class score of the ground-truth classes in y.             #
  ##############################################################################
  pass
  # kenneth yu
  # 1. FF to get scores ,predicted top-1 class,and cache.
  # TODO crop to 10-sub-image ,and average them.

  # 2. set the d_score_predict_class to 1, and other to all 0
  # don calc loss , cause we only care about Score of pred class.We dont care about
  # the softmax (prob) loss.
  # TODO: why not use the predict top-1 label?
  (scores , cache) = model.forward(X, mode='test')
  # use the  input label instead
  # y_pred = np.argmax(scores,axis=1)  # shape (N, )

  dscores = np.zeros_like(scores)  #shape (N,C)
  # use the input label y.
  dscores[np.arange(dscores.shape[0]),y] = 1

  # 3. one BP to get dX.
  dX,_ = model.backward(dscores,cache)  # shape (N,3,H,W)

  # 4. get single value for each pixel: max over color channels for each pixel.
  saliency = np.max(dX, axis=1,keepdims=False)  #shape (N,H,W)
  print 'salience shape:' , saliency.shape


  # 5. TODO: average on 10- sub-images.


  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return saliency

# Once you have completed the implementation in the cell above, run the following to visualize some class
# saliency maps on the validation set of TinyImageNet-100-A.


def show_saliency_maps(mask):
  mask = np.asarray(mask)
  X = data['X_val'][mask]
  y = data['y_val'][mask]

  saliency = compute_saliency_maps(X, y, model)

  for i in xrange(mask.size):
    plt.subplot(2, mask.size, i + 1)
    # add the mean to val data. cause during pre-process we subtract tr,val,test images with mean of tr.
    plt.imshow(deprocess_image(X[i], data['mean_image']))

    plt.axis('off')
    plt.title(data['class_names'][y[i]][0])
    plt.subplot(2, mask.size, mask.size + i + 1)
    plt.title(mask[i])
    # saliency is un-normalized luminance image.change to normalized gray scale.
    # saliency shape (mask.size, 64,64)
    (min_s, max_s) = (np.min(saliency[i]) , np.max(saliency[i]))
    normalized_s_map = (saliency[i] - min_s) * 255 / (max_s - min_s)
    plt.imshow(normalized_s_map,cmap=plt.cm.gray)
    plt.axis('off')
  plt.gcf().set_size_inches(10, 4)
  plt.show()

# Show some random images
mask = np.random.randint(data['X_val'].shape[0], size=5)
show_saliency_maps(mask)

# These are some cherry-picked images that should give good results
show_saliency_maps([128, 3225, 2417, 1640, 4619])


# Fooling Images
# We can also use image gradients to generate "fooling images" as discussed in [2]. Given an image and a target class,
#  we can perform gradient ascent over the image to maximize the target class, stopping when the network classifies
#  the image as the target class. Implement the following function to generate fooling images.
#
# [2] Szegedy et al, "Intriguing properties of neural networks", ICLR 2014



def make_fooling_image(X, target_y, model):
  """
  Generate a fooling image that is close to X, but that the model classifies
  as target_y.

  Inputs:
  - X: Input image, of shape (1, 3, 64, 64)
  - target_y: An integer in the range [0, 100)
  - model: A PretrainedCNN

  Returns:
  - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
  """
  X_fooling = X.copy()
  ##############################################################################
  # TODO: Generate a fooling image X_fooling that the model will classify as   #
  # the class target_y. Use gradient ascent on the target class score, using   #
  # the model.forward method to compute scores and the model.backward method   #
  # to compute image gradients.                                                #
  #                                                                            #
  # HINT: For most examples, you should be able to generate a fooling image    #
  # in fewer than 100 iterations of gradient ascent.                           #
  ##############################################################################
  pass
  # kenneth yu.
  from cs231n.layers import softmax_loss
  # TODO:data pre-processing. scale X_fooling pixel intensity to [0,1] ?
  # TODO: scale to [0,1] is to match the level of BP output?
  # input X is zero-mean pre-processed. shape(1,3,64,64).
  c=1e-5
  l_r = 1e3
  max_iterations = 300
  for t in xrange(max_iterations):
    # optimize objective:  minimize c*|X_fooling -X| + loss(X_fooling)_on_target_y. by [2].
    # |X_fooling - X| is l2-norm, for convenience of direvative we use |X_fooling - X| ^2 because
    # | | is positive so we can use ^2 as a monocular function to represent.
    # FF to get scores. stop when satisfying mis-classifed. max score -> target label.
    (scores, cache) = model.forward(X_fooling, mode='test') # scores shape:(1,C)
    if scores[0].argmax() == target_y:
      print ' *** stopping at iteration: %d predicted class:%d ' % (t, target_y)
      break
    # calc d_loss_on_scores.
    (loss, d_loss_on_scores) = softmax_loss(scores, [target_y])
    if t % 10 == 0:
      print 'iteration %d -- loss of fool image: %d -- predicted class: %d' % (t,loss, scores[0].argmax())
      # print 'd_loss_on_scores: ', d_loss_on_scores

    # BP to get gradients.
    (d_X_fooling, _) = model.backward(d_loss_on_scores, cache)

    # add the regu item: d(c*|X_fooling - X|^2 )/ d(X_fooling)
    grads = d_X_fooling + 2 * c * (X_fooling - X) * 1
    if t % 10 == 0:
      pass
      # print ' d_X_fooling :', d_X_fooling
      # print ' grads: ', grads

    # update X_fooling using SGD.
    X_fooling -= l_r * grads

  # end of loop.

  # TODO: data post-processing and show: Original image X w/ correct label,  adversarial image X_fooling w/ target
  # label, difference (X10 times by shift 128? ) ==> show below.

  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return X_fooling


# Run the following to choose a random validation set image that is correctly classified by the network, and then
# make a fooling image.

# Find a correctly classified validation image
while True:
  i = np.random.randint(data['X_val'].shape[0])

  i = 3940

  X = data['X_val'][i:i+1]
  y = data['y_val'][i:i+1]
  y_pred = model.loss(X)[0].argmax()
  if y_pred == y:
    print 'select x index: ', i
    break

target_y = 67
print 'y_pred:%d, target_y: %d,  ' % (y_pred, target_y)
X_fooling = make_fooling_image(X, target_y, model)

# Make sure that X_fooling is classified as y_target
scores = model.loss(X_fooling)
if scores[0].argmax() != target_y:
  print 'The network is not fooled! pred y: ', scores[0].argmax()
  plt.imshow(deprocess_image(X, data['mean_image']))
  plt.show()
  assert 0
  exit(87)

# Show original image, fooling image, and difference
plt.subplot(1, 3, 1)
# data post processing: add the mean. not renorm.
plt.imshow(deprocess_image(X, data['mean_image']))
plt.axis('off')
plt.title(data['class_names'][y][0])

plt.subplot(1, 3, 2)
# kennet yu:  remember to add the mean to fool image. and re norm pixel intensity to [0,255] cause after
# SGD, the pixel intensity maybe beyond [0,255].
plt.imshow(deprocess_image(X_fooling, data['mean_image'], renorm=True))
plt.title(data['class_names'][target_y][0])
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Difference')
# also add the mean. TODO: enlarge x10?
print ' Difference value:', (X_fooling - X)

plt.imshow(deprocess_image(X - X_fooling, data['mean_image'],renorm=True))
plt.axis('off')
plt.show()
