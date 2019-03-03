# 多分类问题

# 借助对数几率回归可以回答‘是’或‘否’的问题。
# softmax函数返回有N个分量概率的向量，分量之和为1。

# import requests
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 下载数据集，已经下载好，不需要再次下载
# r = requests.get('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

# with open('./data/iris/iris.data', 'w') as f:
#     f.write(r.text)

# 读取csv, 并设置列名
data = pd.read_csv('iris.data', names=['e_cd', 'e_kd', 'b_cd', 'b_kd', 'cat'])

sns.pairplot(data)

# plt.show()

# print(data)

print(data.cat.unique())
# 独热编码
data['c1'] = np.array(data['cat'] == 'Iris-setosa').astype(np.float32)
data['c2'] = np.array(data['cat'] == 'Iris-versicolor').astype(np.float32)
data['c3'] = np.array(data['cat'] == 'Iris-virginica').astype(np.float32)

target = np.stack([data.c1.values, data.c2.values, data.c3.values]).T

# 特征
tz = np.stack([data.e_cd.values, data.e_kd.values, data.b_cd.values, data.b_kd.values]).T

# 定义网络
x = tf.placeholder('float', shape=[None, 4])
y = tf.placeholder('float', shape=[None, 3])

weight = tf.Variable(tf.truncated_normal([4, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

combine_input = tf.matmul(x, weight) + bias

# 预测
pred = tf.nn.softmax(combine_input)

# 计算损失
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=combine_input))

# 正确率, 取向量中最大值的索引
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# 类型转换后求均值
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        index = np.random.permutation(len(target))
        tz = tz[index]
        target = target[index]
        sess.run(train_step, feed_dict={x: tz, y: target})
        if i % 1000 == 0:
            print(sess.run((loss, accuracy), feed_dict={x: tz, y: target}))


