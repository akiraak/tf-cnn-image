# -*- coding: utf-8 -*-
import sys
sys.path.append('tensorflow/tensorflow/examples/tutorials/mnist')
import input_data
import tensorflow as tf
import numpy as np
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


def train():
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


def create_images(tag):
    batch = mnist.train.next_batch(10)
    feed_dict = {x: batch[0], keep_prob: 1.0}

    # 畳み込み１層
    h_conv1_result = h_conv1.eval(feed_dict=feed_dict)
    for i, result in enumerate(h_conv1_result):
        images = channels_to_images(result)
        save_image("3_%s_h_conv1_%02d.png" % (tag, i), images)

    # プーリング１層
    h_pool1_result = h_pool1.eval(feed_dict=feed_dict)
    for i, result in enumerate(h_pool1_result):
        images = channels_to_images(result)
        save_image("3_%s_h_pool1_%02d.png" % (tag, i), images)

    # 畳み込み２層
    h_conv2_result = h_conv2.eval(feed_dict=feed_dict)
    for i, result in enumerate(h_conv2_result):
        images = channels_to_images(result)
        save_image("3_%s_h_conv2_%02d.png" % (tag, i), images)

    # プーリング２層
    h_pool2_result = h_pool2.eval(feed_dict=feed_dict)
    for i, result in enumerate(h_pool2_result):
        images = channels_to_images(result)
        save_image("3_%s_h_pool2_%02d.png" % (tag, i), images)

    print("Created images. tag =", tag)
    print("Number: ", [v.argmax() for v in batch[1]])



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 画像作成
    create_images("before")
    # トレーニング
    train()
    # 画像作成
    create_images("after")
