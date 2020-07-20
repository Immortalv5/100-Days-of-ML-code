# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 22:08:58 2020

@author: Amars
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

N = 1000
num_steps = 800

x_train = np.random.normal(size = N)
m = np.random.normal(loc = 0.5, scale = 0.2, size = N)
b = np.random.normal(loc = 1.0, scale = 0.2, size = N)
y_train = m * x_train + b

x_col = tf.compat.v1.feature_column.numeric_column('x_coords')

estimator = tf.compat.v1.estimator.LinearRegressor([x_col])

train_input = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x = {'x_coords': x_train}, y = y_train,
        shuffle = True, num_epochs = num_steps)

estimator.train(train_input)

predict_input = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x = {'x_coords': np.array([1.0, 2.0], dtype = np.float32)},
        num_epochs = 1, shuffle = False)

results = estimator.predict(predict_input)

for value in results:
    print(value['predictions'])