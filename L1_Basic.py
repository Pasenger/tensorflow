# 机器学习基础
# 线性回归模型

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 从0到10取30个值
data_x = np.linspace(0, 10, 30)

# np.random.normal(0, 1, 30): 噪声，从0到1取30个值
data_y = data_x * 3 + 7 + np.random.normal(0, 1, 30)

# 散点图
plt.scatter(data_x, data_y)
# plt.show()

# 下面通过tensorflow拟合这个模型：y = w * x + b, 目标：求出w和b

# 步骤:
# 1. 定义参数
# 2. 输出训练数据
# 3. 执行推断
# 4. 计算损失：推断与实际情况相比，相差多少
# 5. 【核心】通过训练模型来降低损失，让推断模型与实际值更加接近
# 6. 评估模型，看在实际应用中模型的表现如何

# 步骤1： 定义参数
# 权重
w = tf.Variable(1., name='quanzhong')
# 偏置
b = tf.Variable(0., name='pianzhi')

# 步骤2： 输入数据
# shape为None时，形状可以时任何形状
x = tf.placeholder(tf.float32, shape=None)

# shape为一维任何长度的向量
y = tf.placeholder(tf.float32, shape=[None])

# 步骤3： 推断
pred = tf.multiply(x, w) + b

# 步骤4： 计算损失
# 使用平方差计算, 通过reduce_sum计算损失总和
loss = tf.reduce_sum(tf.squared_difference(pred, y))

# 梯度下降法：让loss降低到最低点，求极值
# 学习速率
learn_rate = 0.0001

# 使用梯度下降学习算法, 最小化loss
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

# 定义session, 训练模型
sess = tf.Session()

# 初始化变量
sess.run(tf.global_variables_initializer())

# 训练10000步
for i in range(10000):
    sess.run(train_step, feed_dict={x: data_x, y: data_y})
    if i % 1000 == 0:
        print(sess.run([loss, w, b], feed_dict={x: data_x, y: data_y}))

sess.close()

# 结果如下： w = 2.9602332, b = 7.1961694
# [6249.381, 1.614691, 0.10198425]
# [50.548775, 3.174291, 5.755518]
# [34.93146, 3.0059364, 6.8885756]
# [34.218925, 2.9699767, 7.1305933]
# [34.186417, 2.9622955, 7.182287]
# [34.18495, 2.9606552, 7.1933265]
# [34.184864, 2.960306, 7.1956763]
# [34.18486, 2.9602332, 7.1961694]
# [34.18486, 2.9602332, 7.1961694]
# [34.18486, 2.9602332, 7.1961694]



