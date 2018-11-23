# coding: utf-8

from __future__ import print_function
import tensorflow as tf
import os


# Declear flags
tf.app.flags.DEFINE_string(
    'log_dir', os.path.join(os.path.dirname(__file__), 'logs'), 'Directory where event logs are written to'
)

# Store flags
FLAGS = tf.app.flags.FLAGS


# constant value
a = tf.constant(5.0, name="a")
b = tf.constant(10.0, name="b")

# basic operations
x = tf.add(a, b, name='add')
y = tf.div(a, b, name='divide')

# run the session
with tf.Session() as sess:

    writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
    print("a = ", sess.run(a))
    print("b = ", sess.run(b))
    print("a + b = ", sess.run(x))
    print("a / b = ", sess.run(y))


# Closing the writer
writer.close()
sess.close()
