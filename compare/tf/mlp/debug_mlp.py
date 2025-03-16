import sys
import os
import pickle
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(cur_dir, '../..'))

sys.path.append(root_path)

import argparse
import tensorflow as tf
import numpy as np
from src.model import mlp


def mlp_compare(x:np.ndarray, n_state:int, hparams:argparse.Namespace=None):
    # 创建 TensorFlow 计算图
    tf.reset_default_graph()  # 清空当前计算图，确保不影响后续运行

    # 1. 定义输入数据
    x_input_np = x
    bs, seq_len, feature_dim = x_input_np.shape  # 获取批次大小、序列长度和特征维度
    print("输入数据形状：", x_input_np.shape)
    print("输入数据：", x_input_np)

    # 2. 构建 TensorFlow 计算图
    with tf.Graph().as_default():
        x_ph = tf.placeholder(tf.float32, shape=[bs, seq_len, feature_dim], name='x_ph')  # 定义占位符
        mlp_output = mlp(x_ph, 'mlp', n_state * 4, hparams=hparams)  # 注意力计算

        init = tf.global_variables_initializer()

        # 3. 运行 TensorFlow Session
        with tf.Session() as sess:
            sess.run(init)  # 初始化所有变量

            # Set the params
            all_vars = tf.global_variables()
            for var in all_vars:
                print("Variable name:", var.name, ", shape:", var.shape.as_list())
                if var.name in params:
                    sess.run(var.assign(params[var.name]))
                    print("Setting variable:", var.name, "to", params[var.name])
            y = sess.run(mlp_output, feed_dict={x_ph: x_input_np})  # 运行计算图
            print("MLP 计算结果：", y)
            return y


if __name__ == "__main__":
    batch_size = 1
    seq_len = 2
    feature_dim = 4
    x = np.random.randn(batch_size, seq_len, feature_dim).astype(np.float32)
    params = {
        'mlp/c_fc/w:0': np.random.randn(batch_size, feature_dim, feature_dim * 4).astype(np.float32),
        'mlp/c_fc/b:0': np.random.randn(feature_dim * 4).astype(np.float32),
        'mlp/c_proj/w:0': np.random.randn(batch_size, feature_dim * 4, feature_dim).astype(np.float32),
        'mlp/c_proj/b:0': np.random.randn(feature_dim).astype(np.float32),
    }
    hparams = argparse.Namespace()
    hparams.n_layer = 1
    hparams.n_head = 2
    hparams.n_embd = feature_dim
    hparams.n_ctx = seq_len
    hparams.n_vocab = 100
    y = mlp_compare(x, feature_dim)
    
    datas = {
        'x': x,
        'params': params,
        'y': y,
        'n_head': hparams.n_head,
        'n_embd': hparams.n_embd,
        'n_layer': hparams.n_layer,
        'n_ctx': hparams.n_ctx,
        'n_vocab': hparams.n_vocab,
    }
    with open(os.path.join(cur_dir, "datas.pkl"), 'wb') as f:
        pickle.dump(datas, f)
        print("数据已保存到：", os.path.join(cur_dir, "datas.pkl"))