# -*- coding: utf-8 -*-
import sys
sys.path.append('tensorflow/tensorflow/examples/tutorials/mnist')
import input_data
import tensorflow as tf
from tools import *


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

"""
第1層
"""
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

"""
第2層
"""
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

"""
全結合層への変換
"""
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

"""
Dropout
"""
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

"""
読み出し層
"""
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

"""
モデルの学習
"""
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

"""
モデルの評価
"""
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch = mnist.train.next_batch(50)
    feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0}

    print("W_conv1: ", W_conv1.eval().shape)
    print("b_conv1: ", b_conv1.eval().shape)
    print("x_image: ", x_image.eval(feed_dict=feed_dict).shape)
    print("h_conv1: ", h_conv1.eval(feed_dict=feed_dict).shape)
    print("h_pool1: ", h_pool1.eval(feed_dict=feed_dict).shape)

    print("W_conv2: ", W_conv2.eval().shape)
    print("b_conv2: ", b_conv2.eval().shape)
    print("h_conv2: ", h_conv2.eval(feed_dict=feed_dict).shape)
    print("h_pool2: ", h_pool2.eval(feed_dict=feed_dict).shape)

    print("W_fc1: ", W_fc1.eval().shape)
    print("b_fc1: ", b_fc1.eval().shape)
    print("h_pool2_flat: ", h_pool2_flat.eval(feed_dict=feed_dict).shape)
    print("h_fc1: ", h_fc1.eval(feed_dict=feed_dict).shape)

    print("h_fc1_drop: ", h_fc1_drop.eval(feed_dict=feed_dict).shape)

    print("W_fc2: ", W_fc2.eval().shape)
    print("b_fc2: ", b_fc2.eval().shape)
    print("y_conv: ", y_conv.eval(feed_dict=feed_dict).shape)
