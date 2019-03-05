# tf.data

# 设计原则
# 1. 快速高效，每秒可以读取13000张照片，支持从云端或多个来源读取，支持并行读取
# 2. 灵活，通过dataset.map函数可以对每条数据做自定义的变换；可以把原本已经写好的Python的读取数据集的函数包含在里面，
#    得到Tensorflow的数据集
# 3. 易用， 有着很好的兼容性，只要生成了数据，就可以直接导入数据而不需要手动构造迭代器

# tf.data使用步骤
# 1. 创建Dataset
# 2. 转化Dataset
# 3. 构建iterator

import tensorflow as tf
import numpy as np
import glob
import time

localtime = time.asctime( time.localtime(time.time()) )
print ("start :", localtime)

image_filenames = glob.glob('E:/tensorflow/catdog/train/*.jpg')

# 乱序， 因为数据集前面全部CAT
image_filenames = np.random.permutation(image_filenames)

# 推导出label列表
labels = list(map(lambda x:float(x.split('\\')[1].split('.')[0] == 'cat'), image_filenames))

# 1. 创建dataset: 文件名和对应的label
dataset = tf.data.Dataset.from_tensor_slices((image_filenames, labels))

# print(dataset)
# <TensorSliceDataset shapes: ((), ()), types: (tf.string, tf.float32)>

# 处理dataset的函数
def _pre_read(image_filename, label):
    image = tf.read_file(image_filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize_images(image, (200, 200))
    image = tf.reshape(image, [200, 200, 1])
    image = tf.image.per_image_standardization(image)
    label = tf.reshape(label, [1])

    return image, label

# 2. 转换dataset, 读取数据
dataset = dataset.map(_pre_read)

# 乱序，参数为buffer, 缓存300张， 对缓存数据shuffle
dataset = dataset.shuffle(256)

# 需要对所有数据循环多少次， repeat有一个默认参数, 默认和-1表示无限循环
dataset = dataset.repeat(5)

# 每次训练的batch
dataset = dataset.batch(64)

# print(dataset)
# <BatchDataset shapes: ((?, 200, 200, 1), (?, 1)), types: (tf.float32, tf.float32)>

# 3. 形成iterator
# make_one_shot_iterator: 每迭代一次输出一个Batch
iterator = dataset.make_one_shot_iterator()

# 获取下一个batch的数据, 包含两部分：图片，label
image_batch, label_batch = iterator.get_next()

# print(image_batch)
# Tensor("IteratorGetNext:0", shape=(?, 200, 200, 1), dtype=float32)

conv2d_1 = tf.contrib.layers.convolution2d(
    image_batch,
    num_outputs=32,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
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
    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
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

conv2d_3 = tf.contrib.layers.convolution2d(
    pool_2,
    num_outputs=64,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
    kernel_size=(5, 5),
    activation_fn=tf.nn.relu,
    stride=(1, 1),
    padding='SAME',
    trainable=True
)

pool_3 = tf.nn.max_pool(
    conv2d_3,
    ksize=[1, 3, 3, 1],
    strides=[1, 2, 2, 1],
    padding='SAME'
)

# print(pool_3.get_shape())
# (?, 25, 25, 64)

# 扁平化pool_3
pool_3_flat = tf.reshape(pool_3, [-1, 25 * 25 * 64])

# 全连接层
fc_1 = tf.contrib.layers.fully_connected(
    pool_3_flat,
    1024,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
    activation_fn=tf.nn.relu
)

fc_2 = tf.contrib.layers.fully_connected(
    fc_1,
    192,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
    activation_fn=tf.nn.relu
)

out_w1 = tf.Variable(tf.truncated_normal([192, 1]))
out_b1 = tf.Variable(tf.truncated_normal([1]))
comb_out = tf.matmul(fc_2, out_w1) + out_b1
pred = tf.sigmoid(comb_out)

# print(pred.get_shape())
# (?, 1)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch, logits=comb_out))

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

predicted = tf.cast(pred > 0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, label_batch), tf.float32))

step = 0
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while True:
        try:
            sess.run(train_step)
            if step % 10 == 0:
                res = sess.run([loss, accuracy])
                print(time.asctime(time.localtime(time.time())), step, res)
                saver.save(sess, './ck/CK_L10_tfdata', global_step=step)
        except tf.errors.OutOfRangeError:
            saver.save(sess, './ck/CK_L10_tfdata', global_step=step)
            print('finish')
            print(step)
            break
        step += 1

finishtime = time.asctime(time.localtime(time.time()))
print ("finish :", finishtime)
