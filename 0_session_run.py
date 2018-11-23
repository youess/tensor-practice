# coding: utf-8

from __future__ import print_function
import tensorflow as tf 


# defining some sentence
welcome = tf.constant("Welcome to tensorflow world!")

# run the session
with tf.Session() as sess:
    print("Output: ", sess.run(welcome))         # binary string

# close the session
sess.close()
