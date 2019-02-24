# 对数几率回归

# 对数几率回归回答是或否的问题

# sigmoid函数： 输入一个值，返回一个从0到1的值

# 对于分类问题使用的损失函数时交叉熵， 交叉熵能够输出一个更大的损失值，从而使梯度下降法做出更大的优化


import tensorflow as tf
import numpy as np
import pandas as pd

data = pd.read_csv('./data/titanic/train.csv')

# 列名
print(data.columns)

# 选出比较重要的数据
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# 对空值使用0填充
data = data.fillna(0)

# print(data)

# 将string转为数值
data['Sex'] = pd.factorize(data.Sex)[0]

# Pclass值为1，2，3，会对机器学习造成困扰，进行独热编码，将三个等级转换为三维，去掉潜在的线性关系
data['p1'] = np.array(data['Pclass'] == 1).astype(np.float32)
data['p2'] = np.array(data['Pclass'] == 2).astype(np.float32)
data['p3'] = np.array(data['Pclass'] == 3).astype(np.float32)
del data['Pclass']
# print(data)

# Embarked独热编码
data['e1'] = np.array(data['Embarked'] == 'S').astype(np.float32)
data['e2'] = np.array(data['Embarked'] == 'C').astype(np.float32)
data['e3'] = np.array(data['Embarked'] == 'Q').astype(np.float32)
del data['Embarked']
# print(data)

# 将数据转换为np.array
# np.stack: 把每一列的数据放到一个array中，如Sex列放入一个数组中
data_data = np.stack(
    [data.Sex.values.astype(np.float32), data.Age.values.astype(np.float32), data.SibSp.values.astype(np.float32),
     data.Parch.values.astype(np.float32), data.Fare.values.astype(np.float32), data.p1.values,
     data.p2.values, data.p3.values, data.e1.values, data.e2.values, data.e3.values]).T

# data.Survived是一个894长度的向量，需要变成894行一列，与特征值对应
# print(np.shape(data_data))
# print(np.shape(data.Survived))
data_target = np.reshape(data.Survived.values.astype(np.float32), (891, 1))
# print(np.shape(data_target))

# 定义网络

# 输入, 不设定一次放多少行，但一定时11列
x = tf.placeholder('float', shape=[None, 11])
# 输出
y = tf.placeholder('float', shape=[None, 1])

# 矩阵相乘，行数必须与前面的列数一样, weight输出值是一个标量值，所以时一列
weight = tf.Variable(tf.random_normal([11, 1]))
bias = tf.Variable(tf.random_normal([1]))

# 定义输出： 矩阵相乘
output = tf.matmul(x, weight) + bias

# 推断: sigmoid输出概率值，大于0.5判断为1， 小于0.5判断为0， 使用tf.cast把布尔值转换为floag32
pred = tf.cast(tf.sigmoid(output) > 0.5, tf.float32)

# 使用交叉熵计算损失， 实际值labels为y, 推断值logits为output, 计算交叉熵会进行sigmoid处理
loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output))

# 梯段下降法训练
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 计算正确率, 判断推断值和实际值是否相等。 reduce_mean：求均值
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    # 尽量乱序输入，没10000步所有数据都进行一次循环。
    # 每100行输入进行
    for n in range(len(data_target) // 100):
        # 乱序
        index = np.random.permutation(len(data_target))

        # 打乱顺序
        data_data = data_data[index]
        data_target = data_target[index]

        batch_xs = data_data[n: n + 100]
        batch_ys = data_target[n: n + 100]

        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    if i % 1000 == 0:
        print(sess.run((loss, accuracy), feed_dict={x: batch_xs, y: batch_ys}))

data_test = pd.read_csv('./data/titanic/test.csv')
data_test = data_test.fillna(0)
data_test['Sex'] = pd.factorize(data_test.Sex)[0]
data_test['p1'] = np.array(data_test['Pclass'] == 1).astype(np.float32)
data_test['p2'] = np.array(data_test['Pclass'] == 2).astype(np.float32)
data_test['p3'] = np.array(data_test['Pclass'] == 3).astype(np.float32)
data_test['e1'] = np.array(data_test['Embarked'] == 'S').astype(np.float32)
data_test['e2'] = np.array(data_test['Embarked'] == 'C').astype(np.float32)
data_test['e3'] = np.array(data_test['Embarked'] == 'Q').astype(np.float32)
test_data = np.stack([data_test.Sex.values.astype(np.float32), data_test.Age.values.astype(np.float32),
                      data_test.SibSp.values.astype(np.float32),
                      data_test.Parch.values.astype(np.float32), data_test.Fare.values.astype(np.float32),
                      data_test.p1.values,
                      data_test.p2.values, data_test.p3.values, data_test.e1.values, data_test.e2.values,
                      data_test.e3.values]).T

test_lable = pd.read_csv('./data/titanic/gender_submission.csv')

test_lable = np.reshape(test_lable.Survived.values.astype(np.float32), (418, 1))

acc = sess.run(accuracy, feed_dict={x: test_data, y: test_lable})
print(acc)

sess.close()
