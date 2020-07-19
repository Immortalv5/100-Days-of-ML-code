# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 22:21:32 2020

@author: Amars
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


tf.compat.v1.disable_eager_execution()


N = 1000
learn_rate = 0.1
batch_size = 40
num_batches = 400


x = np.random.normal(size = N)
m_real = np.random.normal(loc = 0.5, scale = 0.2, size = N)
b_real = np.random.normal(loc = 1, scale = 0.2, size = N)

y = m_real * x + b_real

init = tf.compat.v1.global_variables_initializer()

m = tf.Variable(tf.compat.v1.random_normal([]))
b = tf.Variable(tf.compat.v1.random_normal([]))
gstep = tf.Variable(0, trainable = False)

x_holder = tf.compat.v1.placeholder(tf.float32, shape = batch_size)
y_holder = tf.compat.v1.placeholder(tf.float32, shape = batch_size)


model = m * x_holder + b
loss = tf.reduce_mean(tf.pow(model - y_holder, 2))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learn_rate).minimize(loss, global_step = gstep)

op1 = tf.compat.v1.summary.scalar('m', m)
op2 = tf.compat.v1.summary.scalar('b', b)
merged_op = tf.compat.v1.summary.merge_all()

file_writer = tf.compat.v1.summary.FileWriter('tboard')

with tf.compat.v1.Session() as sess:
    
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for _ in range(num_batches):
        
        x_data = np.empty(batch_size)
        y_data = np.empty(batch_size)
        
        for i in range(batch_size):
            index = np.random.randint(0, N)
            x_data[i] = x[index]
            y_data[i] = y[index]
        
        _, summary, step = sess.run([optimizer, merged_op, gstep], feed_dict = {x_holder: x_data, y_holder: y_data})
        
        file_writer.add_summary(summary, global_step = step)
        file_writer.flush()
        
        sess.run(optimizer, feed_dict = {x_holder: x_data, y_holder: y_data})
        