# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Learning 2 Learn problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import sys

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_dataset

import nn
import numpy as np

_nn_initializers = {
    "w": tf.random_normal_initializer(mean=0, stddev=0.01),
    "b": tf.random_normal_initializer(mean=0, stddev=0.01),
}


def simple():
    """Simple problem: f(x) = x^2."""

    def build():
        """Builds loss graph."""
        x = tf.get_variable(
            "x",
            shape=[],
            dtype=tf.float32,
            initializer=tf.ones_initializer())
        return tf.square(x, name="x_squared")

    return build


def simple_multi_optimizer(num_dims=2):
    """Multidimensional simple problem."""

    def get_coordinate(i):
        return tf.get_variable("x_{}".format(i),
                               shape=[],
                               dtype=tf.float32,
                               initializer=tf.ones_initializer())

    def build():
        coordinates = [get_coordinate(i) for i in xrange(num_dims)]
        x = tf.concat(0, [tf.expand_dims(c, 0) for c in coordinates])
        return tf.reduce_sum(tf.square(x, name="x_squared"))

    return build


def quadratic(batch_size=128, num_dims=10, stddev=0.01, dtype=tf.float32):
    """Quadratic problem: f(x) = ||Wx - y||."""

    def build():
        """Builds loss graph."""

        # Trainable variable.
        x = tf.get_variable(
            "x",
            shape=[batch_size, num_dims],
            dtype=dtype,
            initializer=tf.random_normal_initializer(stddev=stddev))

        x = tf.Print(x, [x], "x=", None, 128 * 10)

        # Non-trainable variables.
        w = tf.get_variable("w",
                            shape=[batch_size, num_dims, num_dims],
                            dtype=dtype,
                            initializer=tf.random_uniform_initializer(),
                            trainable=False)

        w = tf.Print(w, [w], "w=", None, 128 * 10)

        y = tf.get_variable("y",
                            shape=[batch_size, num_dims],
                            dtype=dtype,
                            initializer=tf.random_uniform_initializer(),
                            trainable=False)

        y = tf.Print(y, [y], "y=", None, 128 * 10)

        product = tf.squeeze(tf.batch_matmul(w, tf.expand_dims(x, -1)))

        product = tf.Print(product, [product], "product=", None, 128)

        return tf.reduce_mean(tf.reduce_sum((product - y) ** 2, 1))

    return build

def quadratictest(batch_size=128, num_dims_numsity=800, stddev=0.01, dtype=tf.float32):

    """Quadratic problem: f(x) = ||city  to  vistitor||."""
    def build():
        """Builds loss graph."""
        initX = [[i *1.0  for i in range(0, num_dims_numsity )] for j in range(1, batch_size )]

        # Trainable variable.
        x = tf.get_variable(
            "x",
            dtype=dtype,
            initializer=tf.constant(initX))

        # x = tf.Print(x, [x], "x=", None, batch_size * num_dims_numsity)

        # Non-trainable variables.
        city_x_y = tf.get_variable("city_x_y",
                                   shape=[batch_size, num_dims_numsity, 2],
                                   dtype=dtype,
                                   initializer=tf.random_uniform_initializer(minval=0, maxval=4000),
                                   trainable=False)

        # city_x_y = tf.Print(city_x_y, [city_x_y], "city_x_y=", None, batch_size * num_dims_numsity)

        def my_func_city1(x, city_x_y):

            i = 0
            for i_x in x:
                judges = 1
                for j_x in i_x:
                    if judges != 1 :
                        value1 = np.row_stack((value1, city_x_y[i][j_x]))
                    else:
                        value1 = np.array([city_x_y[i][j_x]])
                        judges = 2

                if i != 0 :
                    # print("value1 =")
                    # print(value1)
                    # print("retvalue =")
                    # print(retValue)
                    retValue = np.row_stack((retValue, [value1]))
                else:
                    retValue = np.array([value1])
                i += 1
            return retValue #will modify in the future

        with tf.variable_scope('hidden89'):
             productresultsrc = tf.py_func(my_func_city1, [x, city_x_y], tf.float32)


        def my_func_city2(x, city_x_y):

            i = 0
            for i_x in x:
                judges = 1
                for j_x in i_x:
                    # print("vlaue %d, %d =" % (i, j_x))
                    # print(city_x_y[i][j_x])

                    if judges > 2:
                        value1 = np.row_stack((value1, city_x_y[i][j_x]))
                    elif judges == 2:
                        value1 = np.array([city_x_y[i][j_x]])
                    elif judges == 1 :
                        lastcity = city_x_y[i][j_x]

                    judges += 1

                value1 = np.row_stack((value1, lastcity))

                if i != 0:
                    retValue = np.row_stack((retValue, [value1]))
                else:
                    retValue = np.array([value1])
                i += 1

            #print ("x2=")
            #print (retValue)

            return retValue #will modify in the future

        with tf.variable_scope('hidden4'):
             productresultdest = tf.py_func(my_func_city2, [x, city_x_y], tf.float32)

        path = (productresultsrc-productresultdest)
        #path = tf.Print(path, [path], "path========")

        reduce_sum1 = tf.reduce_sum(path**2, 2)
        #reduce_sum1 = tf.Print(reduce_sum1, [reduce_sum1],"reduce1_sum=")

        pathValue1 = tf.sqrt(reduce_sum1)
        #pathValue1 = tf.Print(pathValue1, [pathValue1],"pathValue1=")

        pathvalue = tf.reduce_mean(pathValue1)
        #pathvalue = tf.Print(pathvalue, [pathvalue],"pathvalue=")

        #def pathleng(src, dst, city_x_y):
        #    # x will be a numpy array with the contents of the placeholder below

        #    print("city location =")
        #    print(city_x_y)

        #    value = 0
        #    for src_i, dst_j in zip(src, dst):
        #        for src_A, dst_Z in zip(src_i, dst_j):
        #            # print("hello world")
        #            # print("src =")
        #            # print(src_A)
        #            # print("dest = ")
        #            # print( dst_Z )
        #            value = value + np.linalg.norm(src_A-dst_Z)

        #            # print("total =" )
        #            # print(value)
        #            # exit(0)

        #    return value  # will modify in the future

        #with tf.variable_scope('hidden45'):
        #    out = tf.py_func(pathleng, [productresultsrc, productresultdest,city_x_y], tf.float32)

        return pathvalue

    return build


def ensemble(problems, weights=None):
    """Ensemble of problems.

  Args:
    problems: List of problems. Each problem is specified by a dict containing
        the keys 'name' and 'options'.
    weights: Optional list of weights for each problem.

  Returns:
    Sum of (weighted) losses.

  Raises:
    ValueError: If weights has an incorrect length.
  """
    if weights and len(weights) != len(problems):
        raise ValueError("len(weights) != len(problems)")

    build_fns = [getattr(sys.modules[__name__], p["name"])(**p["options"])
                 for p in problems]

    def build():
        loss = 0
        for i, build_fn in enumerate(build_fns):
            with tf.variable_scope("problem_{}".format(i)):
                loss_p = build_fn()
                if weights:
                    loss_p *= weights[i]
                loss += loss_p
        return loss

    return build


def _xent_loss(output, labels):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(output, labels)
    return tf.reduce_mean(loss)


def mnist(layers,  # pylint: disable=invalid-name
          activation="sigmoid",
          batch_size=128,
          mode="train"):
    """Mnist classification with a multi-layer perceptron."""

    if activation == "sigmoid":
        activation_op = tf.sigmoid
    elif activation == "relu":
        activation_op = tf.nn.relu
    else:
        raise ValueError("{} activation not supported".format(activation))

    # Data.
    data = mnist_dataset.load_mnist()
    data = getattr(data, mode)
    images = tf.constant(data.images, dtype=tf.float32, name="MNIST_images")
    images = tf.reshape(images, [-1, 28, 28, 1])
    labels = tf.constant(data.labels, dtype=tf.int64, name="MNIST_labels")

    # Network.
    mlp = nn.MLP(list(layers) + [10],
                 activation=activation_op,
                 initializers=_nn_initializers)
    network = nn.Sequential([nn.BatchFlatten(), mlp])

    def build():
        indices = tf.random_uniform([batch_size], 0, data.num_examples, tf.int64)
        batch_images = tf.gather(images, indices)
        batch_labels = tf.gather(labels, indices)
        output = network(batch_images)
        return _xent_loss(output, batch_labels)

    return build


CIFAR10_URL = "http://www.cs.toronto.edu/~kriz"
CIFAR10_FILE = "cifar-10-binary.tar.gz"
CIFAR10_FOLDER = "cifar-10-batches-bin"


def _maybe_download_cifar10(path):
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, CIFAR10_FILE)
    if not os.path.exists(filepath):
        print("Downloading CIFAR10 dataset to {}".format(filepath))
        url = os.path.join(CIFAR10_URL, CIFAR10_FILE)
        filepath, _ = urllib.request.urlretrieve(url, filepath)
        statinfo = os.stat(filepath)
        print("Successfully downloaded {} bytes".format(statinfo.st_size))
        tarfile.open(filepath, "r:gz").extractall(path)


def cifar10(path,  # pylint: disable=invalid-name
            conv_channels=None,
            linear_layers=None,
            batch_norm=True,
            batch_size=128,
            num_threads=4,
            min_queue_examples=1000,
            mode="train"):
    """Cifar10 classification with a convolutional network."""

    # Data.
    _maybe_download_cifar10(path)

    # Read images and labels from disk.
    if mode == "train":
        filenames = [os.path.join(path,
                                  CIFAR10_FOLDER,
                                  "data_batch_{}.bin".format(i))
                     for i in xrange(1, 6)]
    elif mode == "test":
        filenames = [os.path.join(path, "test_batch.bin")]
    else:
        raise ValueError("Mode {} not recognised".format(mode))

    depth = 3
    height = 32
    width = 32
    label_bytes = 1
    image_bytes = depth * height * width
    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, record = reader.read(tf.train.string_input_producer(filenames))
    record_bytes = tf.decode_raw(record, tf.uint8)

    label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    raw_image = tf.slice(record_bytes, [label_bytes], [image_bytes])
    image = tf.cast(tf.reshape(raw_image, [depth, height, width]), tf.float32)
    # height x width x depth.
    image = tf.transpose(image, [1, 2, 0])
    image = tf.div(image, 255)

    queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3 * batch_size,
                                  min_after_dequeue=min_queue_examples,
                                  dtypes=[tf.float32, tf.int32],
                                  shapes=[image.get_shape(), label.get_shape()])
    enqueue_ops = [queue.enqueue([image, label]) for _ in xrange(num_threads)]
    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

    # Network.
    def _conv_activation(x):  # pylint: disable=invalid-name
        return tf.nn.max_pool(tf.nn.relu(x),
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME")

    conv = nn.ConvNet2D(output_channels=conv_channels,
                        kernel_shapes=[5],
                        strides=[1],
                        paddings=[nn.SAME],
                        activation=_conv_activation,
                        activate_final=True,
                        initializers=_nn_initializers,
                        use_batch_norm=batch_norm)

    if batch_norm:
        linear_activation = lambda x: tf.nn.relu(nn.BatchNorm()(x))
    else:
        linear_activation = tf.nn.relu

    mlp = nn.MLP(list(linear_layers) + [10],
                 activation=linear_activation,
                 initializers=_nn_initializers)
    network = nn.Sequential([conv, nn.BatchFlatten(), mlp])

    def build():
        image_batch, label_batch = queue.dequeue_many(batch_size)
        label_batch = tf.reshape(label_batch, [batch_size])

        output = network(image_batch)
        return _xent_loss(output, label_batch)

    return build
