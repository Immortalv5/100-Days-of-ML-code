# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 23:02:40 2020

@author: Amars
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.compat.v1.app.flags.FLAGS

tf.compat.v1.app.flags.DEFINE_string(
    'train_file', 'mnist_train.tfrecords', 'Training file')
tf.compat.v1.app.flags.DEFINE_string(
    'test_file', 'mnist_test.tfrecords', 'Testing file')
tf.compat.v1.app.flags.DEFINE_string(
    'out_dir', 'dnn_output', 'Output directory')
tf.compat.v1.app.flags.DEFINE_integer(
    'train_steps', 8000, 'Number of training steps')
tf.compat.v1.app.flags.DEFINE_integer(
    'batch_size', 80, 'Number of images in a batch')
tf.compat.v1.app.flags.DEFINE_integer(
    'test_steps', 100, 'Number of tests to perform')

tf.compat.v1.disable_eager_execution()

image_dim = 28
num_labels = 10
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
    dataset = tf.compat.v1.data.TFRecordDataset(FLAGS.train_file)
    dataset = dataset.map(parser).repeat().batch(FLAGS.batch_size)
    image, label = dataset.make_one_shot_iterator().get_next()
    return {'pixels': image}, label

def test_func():
    dataset = tf.compat.v1.data.TFRecordDataset(FLAGS.test_file)
    dataset = dataset.map(parser).repeat().batch(FLAGS.batch_size)
    image, label = dataset.make_one_shot_iterator().get_next()
    return {'pixels': image}, label

train_spec = tf.compat.v1.estimator.TrainSpec(train_func, max_steps = FLAGS.train_steps)
eval_spec = tf.compat.v1.estimator.EvalSpec(test_func, steps = FLAGS.test_steps)

tf.compat.v1.estimator.train_and_evaluate(dnn_classifier, train_spec, eval_spec)
