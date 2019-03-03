import tensorflow as tf
import glob
import os

# 形成文件名列表
image_filenames = glob.glob('E:/tensorflow/catdog/train/*.jpg')

# 推导出label列表
labels = list(map(lambda x:x.split('\\')[1].split('.')[0], image_filenames))

# 独热编码
train_label = [[1, 0] if x == 'cat' else [0, 1] for x in labels]

# 对照片目录的队列进行读取，形成了两个队列: 文件名、label
image_queue = tf.train.slice_input_producer([image_filenames, train_label])

# 解码
image_ = tf.read_file(image_queue[0])
image = tf.image.decode_jpeg(image_, channels=3)

# 将照片转换为灰度照片，channels = 1, 加快计算速度
grey_image = tf.image.rgb_to_grayscale(image)
# 统一照片大小
resize_image = tf.image.resize_images(grey_image, (200, 200))
# reshape
resize_image = tf.reshape(resize_image, [200, 200, 1])

# 标准化，加快CNN网络的训练过程
new_image = tf.image.per_image_standardization(resize_image)

# batch
batch_size = 60

# 形成batch使用， 一次读入130张， 加快速度
capacity = 10 + 2 * batch_size

# 形成批次
image_batch, label_batch = tf.train.batch([new_image, image_queue[1]], batch_size=batch_size, capacity=capacity)

# 第一层
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

pool_1 = tf.nn.max_pool(conv2d_1,
                        # [0]：batch; [3]: channel, 一般设置为1， [1]\[2]: 对哪个部位进行运算
                        ksize=[1, 3, 3, 1],
                        # 跨度: 2 * 2
                        strides=[1, 2, 2, 1],
                        padding='SAME')

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

pool_2 = tf.nn.max_pool(conv2d_2,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')

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

pool_3 = tf.nn.max_pool(conv2d_3,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')

# 扁平化
pool3_flat = tf.reshape(pool_3, [-1, 25 * 25 * 64])

fc_1 = tf.contrib.layers.fully_connected(
    pool3_flat,
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

out_w1 = tf.Variable(tf.truncated_normal([192, 2]))
out_b1 = tf.Variable(tf.truncated_normal([2]))
comb_out = tf.matmul(fc_2, out_w1) + out_b1
pred = tf.sigmoid(comb_out)

label_batch = tf.cast(label_batch, tf.float32)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch, logits=comb_out))

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

predicted = tf.cast(pred > 0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, label_batch), tf.float32))

ckpt = tf.train.get_checkpoint_state(os.path.dirname('./ck/'))
saver = tf.train.Saver()
sess = tf.Session()
# 初始化参数
sess.run(tf.global_variables_initializer())
saver.restore(sess, ckpt.model_checkpoint_path)

# 控制队列
coord = tf.train.Coordinator()
# 启动队列
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# 训练3000步:
for step in range(2000, 3000):
    sess.run(train_step)
    if step % 100 == 0:
        res = sess.run([loss, accuracy])
        print(step, res)
        saver.save(sess, './ck/CK_L8_catdog', global_step=step)
coord.request_stop()
coord.join(threads)

sess.close()
