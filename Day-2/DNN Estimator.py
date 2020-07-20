# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 22:08:29 2020

@author: Amars
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

image_dim = 28
num_labels = 10
batch_size = 80
num_steps = 8000
hidden_layers = [128, 32]

def parser(example):
    features = tf.compat.v1.parse_single_example(example, 
                features = {'images': tf.compat.v1.FixedLenFeature([], tf.string),
                'labels': tf.compat.v1.FixedLenFeature([], tf.int64)})
    pixels = tf.compat.v1.decode_raw(features['images'], tf.uint8)
    pixels.set_shape([image_dim * image_dim])
    pixels = tf.compat.v1.cast(pixels, tf.float32) * (1.0/255) - 0.5
    labels = features['labels']
    return pixels, labels

column = tf.compat.v1.feature_column.numeric_column('pixels', shape = [image_dim * image_dim])

dnn_classifier = tf.compat.v1.estimator.DNNClassifier(hidden_layers, [column],
                        model_dir = 'dnn_output', n_classes = num_labels)

def train_func():
    dataset = tf.compat.v1.data.TFRecordDataset('mnist_train.tfrecords')
    dataset = dataset.map(parser).repeat().batch(batch_size)
    image, label = dataset.make_one_shot_iterator().get_next()
    return {'pixels': image}, label

dnn_classifier.train(train_func, steps = num_steps)

def test_func():
    dataset = tf.compat.v1.data.TFRecordDataset('mnist_test.tfrecords')
    dataset = dataset.map(parser).repeat().batch(batch_size)
    image, label = dataset.make_one_shot_iterator().get_next()
    return {'pixels': image}, label

metrics = dnn_classifier.evaluate(test_func)

for key, value in metrics.items():
    print(key, ':', value)