# tensorflow的读取机制

# 机制一

# 通过输入feed_dict参数配合placeholder, 可以启动运算过程

# 机制二
# 文件读取管线，步骤：
# 1. 生成文件名列表glob或者tf.train.match_filenames_once
# 2. 生成对列名列表
#       tf.train.string_input_producer: 形成一个队列
#       tf.train.slice_input_producer：可以形成两个队列(可配置参数设置文件名乱序和最大的训练迭代数epoch)
# 3. 针对输入文件格式的阅读器
# 4. 形成批次(batch)
# 5. 启动队列


import numpy as np

