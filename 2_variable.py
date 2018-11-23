# coding: utf-8

from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops


# Variable

# 创建几个有默认值的变量
weights = tf.Variable(tf.random_normal([2, 3], stddev=0.1, name="weights"))
biases = tf.Variable(tf.zeros([3]), name="bias")
custom_variable = tf.Variable(tf.zeros([3]), name="custom")

# 收集所有变量的张量，并存于一个list里面
all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)


# Custom initializer

# 选择需要初始化的变量
variable_list_custom = [weights, custom_variable]

# 初始化
init_custom_op = tf.variables_initializer(var_list=variable_list_custom)


# Global initializer

# 方法1
init_all_op = tf.global_variables_initializer()

# 方法2
init_all_op = tf.variables_initializer(var_list=all_variables_list)


# Initialization using other variables

# 创建
WeightsNew = tf.Variable(weights.initialized_value(), name="WeightsNew")

# 初始化
init_WeightsNew_op = tf.variables_initializer(var_list=[WeightsNew])


# Running the session
with tf.Session() as sess:

    # 错误，还没有执行初始化操作
    # print("variables: ", sess.run(weights))

    # 会话里执行初始化动作
    sess.run(init_custom_op)
    sess.run(init_all_op)
    sess.run(init_WeightsNew_op)

    print("variables: ", sess.run(weights))

    print("new weights: ", sess.run(WeightsNew))
