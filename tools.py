# -*- coding: utf-8 -*-
import math
import tensorflow as tf
import numpy as np
from PIL import Image


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def save_image(file_name, image_ndarray, cols=8):
    # 画像数, 幅, 高さ
    count, w, h = image_ndarray.shape
    # 縦に画像を配置する数
    rows = int((count - 1) / cols) + 1
    # 復数の画像を大きな画像に配置し直す
    canvas = Image.new("RGB", (w * cols + (cols - 1), h * rows + (rows - 1)), (0x80, 0x80, 0x80))
    for i, image in enumerate(image_ndarray):
        # 横の配置座標
        x_i = int(i % cols)
        x = int(x_i * w + x_i * 1)
        # 縦の配置座標
        y_i = int(i / cols)
        y = int(y_i * h + y_i * 1)
        out_image = Image.fromarray(np.uint8(image))
        canvas.paste(out_image, (x, y))
    canvas.save('images/' + file_name, "PNG")


def channels_to_images(channels):
    count = channels.shape[2]
    images = []
    for i in range(count):
        image = []
        for line in channels:
            out_line = [pix[i] for pix in line]
            image.append(out_line)
        images.append(image)
    return np.array(images) * 255
