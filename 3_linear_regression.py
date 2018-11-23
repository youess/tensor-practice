# coding: utf-8


import numpy as np
import tensorflow as tf
import xlrd
import matplotlib.pyplot as plt
import os
from sklearn.utils import check_random_state


# 创建人工数据, 单变量线性回归
n = 50
XX = np.arange(n)
rs = check_random_state(0)
YY = rs.randint(-20, 20, size=(n, )) + 2.0 * XX
data = np.stack([XX, YY], axis=1)
print(data.shape)


# flags
tf.app.flags.DEFINE_integer('num_epochs', 50, 'The number of epochs for training the model. Default=50')

FLAGS = tf.app.flags.FLAGS


# create weights and bias
W = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")


# Create placeholders for input X and label Y
def inputs():
    """
    定义占位符，并返回数据和标签的占位符
    """
    X = tf.placeholder(tf.float32, name="X")
    Y = tf.placeholder(tf.float32, name="Y")
    return X, Y


# 预测
def inference(X):
    """
    前向传播，输入X，输出预测Y
    """
    return X * W + b


# 损失函数
def loss(X, Y):
    """
    根据预测函数得到的Y_hat与真实Y进行比较
    """
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted)) / (2 * data.shape[0])


# 训练函数
def train(loss):
    """
    传入损失值，然后通过优化算子减小损失
    """
    learning_rate = 1e-4
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:

    # 初始化变量w, b
    sess.run(tf.global_variables_initializer())

    # 获取输入tensor
    X, Y = inputs()

    # 获取训练损失以及train_op
    train_loss = loss(X, Y)
    train_op = train(train_loss)

    # 开始训练模型
    for epoch_num in range(FLAGS.num_epochs):
        loss_value, _ = sess.run([train_loss, train_op], feed_dict={X: data[:, 0], Y: data[:, 1]})

        # print loss per epoch
        print('epoch %d: loss=%.4f' % (epoch_num+1, loss_value))

        # save the values of weights and bias
        wcoeff, bias = sess.run([W, b])


print(f"weights: {wcoeff}, bias: {bias}")

# 画出训练图像
# Input_values = data[:, 0]
# Labels = data[:, 1]
# Prediction_values = data[:, 0] * wcoeff + bias

# plt.plot(Input_values, Labels, 'ro', label='main')
# plt.plot(Input_values, Prediction_values, label='Predicted')

# # Saving the result.
# plt.legend()
# plt.savefig('plot.png')
# plt.close()