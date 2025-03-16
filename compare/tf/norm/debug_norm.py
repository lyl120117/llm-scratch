import sys
import os
import pickle
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(cur_dir, '../..'))

sys.path.append(root_path)
import tensorflow as tf
import numpy as np
from src.model import norm


def norm_compare(x:np.ndarray, params:dict):
    # 创建 TensorFlow 计算图
    tf.reset_default_graph()  # 清空当前计算图，确保不影响后续运行

    # 1. 定义输入数据
    x_input_np = x
    feature_dim = x_input_np.shape[-1]  # 获取特征维度
    print("输入数据形状：", x_input_np.shape)
    print("输入数据：", x_input_np)

    # 2. 构建 TensorFlow 计算图
    with tf.Graph().as_default():
        x_ph = tf.placeholder(tf.float32, shape=[None, None, feature_dim], name='x_ph')  # 定义占位符
        normed_x = norm(x_ph, scope='norm_layer')  # 归一化

        init = tf.global_variables_initializer()

        # 3. 运行 TensorFlow Session
        with tf.Session() as sess:
            sess.run(init)  # 初始化所有变量

            # Set the params
            all_vars = tf.global_variables()
            for var in all_vars:
                if var.name in params:
                    print("Setting variable:", var.name, "to", params[var.name])
                    sess.run(var.assign(params[var.name]))
            result = sess.run(normed_x, feed_dict={x_ph: x_input_np})  # 运行计算图
            print("归一化后的结果：", result)

            return result


if __name__ == "__main__":
    batch_size = 2
    seq_len = 6
    feature_dim = 48
    x = np.random.randn(batch_size, seq_len, feature_dim).astype(np.float32)
    params = {
        'norm_layer/g:0': np.random.randn(feature_dim).astype(np.float32),
        'norm_layer/b:0': np.random.randn(feature_dim).astype(np.float32),
    }
    y = norm_compare(x, params)
    
    datas = {
        'x': x,
        'params': params,
        'y': y
    }

    with open(os.path.join(cur_dir, "datas.pkl"), 'wb') as f:
        pickle.dump(datas, f)
        print("数据已保存到：", os.path.join(cur_dir, "datas.pkl"))