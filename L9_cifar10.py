# cirar10数据集

import numpy as np 
import tensorflow as tf

x = tf.placeholder('float', shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.int32, shape=[None])
keep_porb = tf.placeholder('float')

def img_pre(image):
    # 随机调节图像的亮度
    new_img = tf.image.random_brightness(image, max_delta=63)
    # 随机左右翻转图像
    new_img = tf.image.random_flip_left_right(new_img)
    # 随机调整图像对比度
    new_img = tf.image.random_contrast(new_img, lower=0.2, upper=1.8)
    # 标准化，将图像的所有特征都规范于0~1的范围内
    new_img = tf.image.per_image_standardization(new_img)

    return new_img

img = tf.map_fn(img_pre, x)

batch_size = 128
learning_rate = 0.0001

# 正则函数
reg = tf.contrib.layers.l2_regularizer(scale=0.1)

conv2d_1 = tf.contrib.layers.convolution2d(
    img,
    num_outputs=32,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.0001),
    weights_regularizer=reg,
    kernel_size=(5, 5),
    activation_fn=tf.nn.relu,
    stride=(1, 1),
    padding='SAME',
    trainable=True
)

pool_1 = tf.nn.max_pool(
    conv2d_1,
    ksize=[1, 3, 3, 1],
    strides=[1, 2, 2, 1],
    padding='SAME'
)

conv2d_2 = tf.contrib.layers.convolution2d(
    pool_1,
    num_outputs=32,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
    weights_regularizer=reg,
    kernel_size=(5, 5),
    activation_fn=tf.nn.relu,
    stride=(1, 1),
    padding='SAME',
    trainable=True
)

pool_2 = tf.nn.max_pool(
    conv2d_2,
    ksize=[1, 3, 3, 1],
    strides=[1, 2, 2, 1],
    padding='SAME'
)

#[5, 5, 32, 64]：　跨度，跨度，　输入，　输出
conv_2d_w3 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.01))
conv_2d_b3 = tf.Variable(tf.truncated_normal([64]))
conv_2d_3 = tf.nn.conv2d(pool_2, conv_2d_w3, strides=[1, 1, 1, 1], padding='SAME') + conv_2d_b3
conv_2d_3_output = tf.nn.relu(conv_2d_3)

pool_3 = tf.nn.max_pool(
    conv_2d_3_output,
    ksize=[1, 3, 3, 1],
    strides=[1, 2, 2, 1],
    padding='SAME'
)

pool_3_flat = tf.reshape(pool_3, [-1, 4 * 4 * 64])
fc_1 = tf.contrib.layers.fully_connected(
    pool_3_flat,
    1024,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
    weights_regularizer=reg,
    activation_fn=tf.nn.relu
)

fc_2 = tf.contrib.layers.fully_connected(
    fc_1,
    128,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
    weights_regularizer=reg,
    activation_fn=tf.nn.relu
)

# dropout
fc2_drop = tf.nn.dropout(fc_2, keep_porb)

# softmax
out_w1 = tf.Variable(tf.truncated_normal([128, 10]))
out_b1 = tf.Variable(tf.truncated_normal([10]))
combine = tf.matmul(fc2_drop, out_w1) + out_b1

# 预测结果
pred = tf.cast(tf.argmax(tf.nn.softmax(combine), 1), tf.int32)

# 应用正则化方法
weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

# 对所有可训练的参数应用正则函数
reg_ws = tf.contrib.layers.apply_regularization(reg, weights_list=weights)

# 计算loss, 没有使用独热编码所以使用sparse_softmax_cross_entropy_with_logits
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=combine))

# loss加上应用正则化后的值作为loss
loss_fn = loss + tf.reduce_sum(reg_ws)

# 训练
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_fn)

# 正确率
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, y_), tf.float32))

# 预处理cifar10数据
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_list = []
label_list = []

for i in range (1, 6):
    data = unpickle('E:/tensorflow/cifar-10-batches-py/data_batch_{}'.format(i))
    data_list.append(data[b'data'])
    label_list.append(data[b'labels'])

all_data = np.concatenate(data_list)
all_label = np.concatenate(label_list)

def generatebatch(X, Y, n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i * batch_size
        end = start + batch_size
        batch_xs = X[start: end]
        batch_ys = Y[start: end]
        # 生成每一个batch
        yield batch_xs, batch_ys

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ii = 0
for epoch in range(0, 100):
    index = np.random.permutation(all_label.shape[0])
    all_data = all_data[index]
    all_label = all_label[index]
    
    for batch_xs, batch_ys in generatebatch(all_data, all_label, all_label.shape[0], batch_size):
        batch_xs = np.array(list(map(lambda x:x.reshape([3, 1024]).T.reshape([32, 32, 3]), batch_xs)))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_porb: 0.5})
        if ii % 100 == 0:
            print(sess.run([loss, accuracy, ], feed_dict={x: batch_xs, y_: batch_ys, keep_porb: 1}))
        ii += 1
    if epoch % 2 == 0:
        res = sess.run([loss, accuracy, ], feed_dict={x: batch_xs, y_: batch_ys, keep_porb: 1})
        print(epoch, res)
        saver.save(sess, './ck/L9_SIFAR10', global_step=epoch)

test = unpickle('E:/tensorflow/cifar-10-batches-py/test_batch')
test_label_hot = test[b'labels']
test_data = test[b'data']

righ = []
for batch_xs, batch_ys in generatebatch(test_data, test_label_hot, test_data.shape[0], 128):
    batch_xs = np.array(list(map(lambda x:x.reshape([3, 1024]).T.reshape([32, 32, 3]), batch_xs)))
    acc = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_porb: 1})
    righ.append(acc)

print(sess.run(tf.reduce_mean(righ)))
# 0.7701322

sess.close()
