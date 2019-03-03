import numpy as np
import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/MNIST_data/', one_hot=True)

x = tf.placeholder('float', shape=[None, 784])
# label长度为10
y_ = tf.placeholder('float', shape=[None, 10])

# 使用4个维度表示一张图片：批次、高度、宽度、channel
x_image = tf.reshape(x, [-1, 28, 28, 1])

conv2d_1 = tf.contrib.layers.convolution2d(
    x_image,
    num_outputs=32,
    kernel_size=(5, 5),
    activation_fn=tf.nn.relu,
    stride=(1, 1),
    padding='SAME',
    trainable=True)

# 降采样
pool_1 = tf.nn.max_pool(conv2d_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(conv2d_1.get_shape())
print(pool_1.get_shape())

conv2d_2 = tf.contrib.layers.convolution2d(
    pool_1,
    num_outputs=64,
    kernel_size=(5, 5),
    activation_fn=tf.nn.relu,
    stride=(1, 1),
    padding='SAME',
    trainable=True)

# 降采样
pool_2 = tf.nn.max_pool(conv2d_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(conv2d_2.get_shape())
print(pool_2.get_shape())

# 全连接层
# 扁平化pool_2, 参数1是批次，不知道，所以写-1
pool2_flat = tf.reshape(pool_2, [-1, 7 * 7 * 64])

fc_1 = tf.contrib.layers.fully_connected(pool2_flat, 1024, activation_fn=tf.nn.relu)

# dropout：随机丢弃一些单元, 可以减轻网络的过拟合问题
# keep_prob:保存多少
keep_prob = tf.placeholder('float')
fc1_drop = tf.nn.dropout(fc_1, keep_prob)

# 全连接分类
fc_2 = tf.contrib.layers.fully_connected(
    fc1_drop,
    10,
    activation_fn=tf.nn.softmax
)

loss = -tf.reduce_mean(y_ * tf.log(fc_2))

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    if i % 1000 == 0:
        print(sess.run(loss, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1}))

correct_prediction = tf.equal(tf.argmax(fc_2, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1})

print('acc: ', acc)

sess.close()
