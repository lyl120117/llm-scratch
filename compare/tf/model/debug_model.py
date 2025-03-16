import sys
import os
import pickle
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(cur_dir, '../..'))

sys.path.append(root_path)

import argparse
import tensorflow as tf
import numpy as np
from src.model import model


def model_compare(x:np.ndarray, hparams:argparse.Namespace=None, past:np.ndarray=None, params:dict=None):
    # 创建 TensorFlow 计算图
    tf.reset_default_graph()  # 清空当前计算图，确保不影响后续运行

    # 1. 定义输入数据
    x_input_np = x
    bs, seq_len = x_input_np.shape  # 获取批次大小、序列长度和特征维度
    print("输入数据形状：", x_input_np.shape)
    print("输入数据：", x_input_np)

    # 2. 构建 TensorFlow 计算图
    with tf.Graph().as_default():
        x_ph = tf.placeholder(tf.int32, shape=[bs, seq_len], name='x_ph')  # 定义占位符
        if past is not None:
            past_ph = tf.placeholder(tf.float32, shape=[bs, hparams.n_layer, 2, hparams.n_head, seq_len, hparams.n_embd // hparams.n_head], name='past_ph')
            model_output = model(X=x_ph, past=past_ph, scope='model', hparams=hparams)  # 注意力计算
        else:
            model_output = model(X=x_ph, scope='model', hparams=hparams)

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
            ret = sess.run(model_output, feed_dict={x_ph: x_input_np, past_ph: past})
            y = ret['logits']
            present = ret['present']
            print("模型计算结果：", y)
            print("Present 计算结果：", present)
            return y, present


if __name__ == "__main__":
    batch_size = 3
    seq_len = 10
    feature_dim = 64
    hparams = argparse.Namespace()
    hparams.n_layer = 1
    hparams.n_head = 4
    hparams.n_embd = feature_dim
    hparams.n_ctx = 50
    hparams.n_vocab = 100

    x = np.random.randint(0, hparams.n_vocab, size=(batch_size, seq_len)).astype(np.int32)
    params = {
        'model/wpe:0': np.random.randn(hparams.n_ctx, hparams.n_embd).astype(np.float32),
        'model/wte:0': np.random.randn(hparams.n_vocab, hparams.n_embd).astype(np.float32),
        'model/h0/ln_1/g:0': np.random.randn(feature_dim).astype(np.float32),
        'model/h0/ln_1/b:0': np.random.randn(feature_dim).astype(np.float32),
        'model/h0/attn/c_attn/w:0': np.random.randn(1, feature_dim, feature_dim * 3).astype(np.float32),
        'model/h0/attn/c_attn/b:0': np.random.randn(feature_dim * 3).astype(np.float32),
        'model/h0/attn/c_proj/w:0': np.random.randn(1, feature_dim, feature_dim).astype(np.float32),
        'model/h0/attn/c_proj/b:0': np.random.randn(feature_dim).astype(np.float32),
        'model/h0/ln_2/g:0': np.random.randn(feature_dim).astype(np.float32),
        'model/h0/ln_2/b:0': np.random.randn(feature_dim).astype(np.float32),
        'model/h0/mlp/c_fc/w:0': np.random.randn(1, feature_dim, feature_dim * 4).astype(np.float32),
        'model/h0/mlp/c_fc/b:0': np.random.randn(feature_dim * 4).astype(np.float32),
        'model/h0/mlp/c_proj/w:0': np.random.randn(1, feature_dim * 4, feature_dim).astype(np.float32),
        'model/h0/mlp/c_proj/b:0': np.random.randn(feature_dim).astype(np.float32),
        'model/ln_f/g:0': np.random.randn(feature_dim).astype(np.float32),
        'model/ln_f/b:0': np.random.randn(feature_dim).astype(np.float32),
    }
    past = np.random.randn(batch_size, hparams.n_layer, 2, hparams.n_head, seq_len, hparams.n_embd // hparams.n_head).astype(np.float32)
    print('past shape:', past.shape)
    y, present = model_compare(x=x, past=past, params=params, hparams=hparams)
    
    datas = {
        'x': x,
        'params': params,
        'y': y,
        'n_head': hparams.n_head,
        'n_embd': hparams.n_embd,
        'n_layer': hparams.n_layer,
        'n_ctx': hparams.n_ctx,
        'n_vocab': hparams.n_vocab,
        'past': past,
        'past_shape': past.shape,
        'present': present,
    }
    with open(os.path.join(cur_dir, "datas.pkl"), 'wb') as f:
        pickle.dump(datas, f)
        print("数据已保存到：", os.path.join(cur_dir, "datas.pkl"))