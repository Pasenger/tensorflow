import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/MNIST_data/', one_hot=True)

# print(len(mnist.train.images), len(mnist.train.labels))
# print(len(mnist.test.images), len(mnist.test.labels))
#
# print(mnist.train.images[0])
# print(len(mnist.train.images[0]))
#
# plt.imshow(mnist.train.images[1].reshape(28, 28))
# plt.show()
# print(mnist.train.labels[1])

# 使用softmax进行分类

x = tf.placeholder('float', shape=[None, 784])
# label长度为10
y_ = tf.placeholder('float', shape=[None, 10])

weight = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, weight) + b)

# loss使用交叉熵的定义进行运算
loss = -tf.reduce_mean(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1100):
    batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
    if i % 100 == 0:
        print(sess.run(loss, feed_dict={x: batch[0], y_: batch[1]}))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

print('acc: ', acc)

sess.close()
