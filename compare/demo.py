import fire
import tensorflow as tf
import os
import json
import numpy as np

import model, sample, encoder


def main(model_name='124M', seed=None, nsamples=0, batch_size=1,
         length=None, temperature=1, top_k=0, top_p=1,
         models_dir='models'):
    print('Model name:', model_name, ', seed:', seed, ', nsamples:', nsamples, ', batch size:', batch_size,
          ', length:', length, ', temperature:', temperature, ', top_k:', top_k, ', top_p:', top_p,
          ', models dir:', models_dir)
    model_dir = os.path.expanduser(os.path.expandvars(models_dir))

    enc = encoder.get_encoder(model_name, model_dir)
    print('Encoder loaded:', enc)
    text = "Hello, how are you?"
    print('Input text:', text)
    embedding = enc.encode(text)
    print('Encoded text:', embedding, ', type:', type(embedding))
    out_text = enc.decode(embedding)
    print('Decoded text:', out_text)

    hparams = model.default_hparams()
    print('Hyperparameters default:', hparams)
    with open(os.path.join(model_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    print('Hyperparameters loaded:', hparams)

    if length is None:
        length = 1
    print('Length:', length)
    print('Batch size:', batch_size)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        print('Context:', context)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )
        print('Output:', output)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(model_dir, model_name))
        print('Checkpoint:', ckpt, ', type:', type(ckpt))
        saver.restore(sess, ckpt)
        print('Model restored from checkpoint.')

        all_vars = tf.global_variables()
        output_dir = os.path.join('outputs', model_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            for var in all_vars:
                torch_name = var.name.replace(':0', '')
                torch_name = torch_name.replace('model/h', 'models/')
                torch_name = torch_name.replace('model/', '')
                torch_name = torch_name.replace('/', '.')
                dst_file = os.path.join(output_dir, torch_name + '.npy')
                np.save(dst_file, var.eval())
                print("Variable name:", var.name, ', torch_name:', torch_name, ', shape:', var.shape.as_list())

        # Traversal all the variables, and convert them to dict
        # all_vars = tf.global_variables()
        # all_vars_dict = {}
        # for var in all_vars:
        #     all_vars_dict[var.name] = var.eval().tolist()
        #     # print('Variable:', var.name, ', shape:', var.shape.as_list())
        #     all_vars_dict[var.name] = var.eval()
        # print('model/h0/ln_1/b:0:', all_vars_dict['model/h0/ln_1/b:0'])
        # print('model/h0/attn/c_proj/b:0:', all_vars_dict['model/h0/attn/c_proj/b:0'])
        # print('All variables:', all_vars_dict)
        # json.dump(all_vars_dict, open('model.json', 'w'), indent=4)

        raw_text = text
        print('Raw text:', raw_text)
        context_tokens = enc.encode(raw_text)
        print('Context tokens:', context_tokens)

        out = sess.run(output, feed_dict={
            context: [context_tokens for _ in range(batch_size)]
        })
        print('Generated output:', out)
        # [[15496    11   703   389   345    30 21636 27094]]
        out = out[0, len(context_tokens):]
        print('Trimmed output:', out)
        out_text = enc.decode(out)
        print('Decoded output:', out_text)
        

# python src/demo.py --seed=199 --model_name=1558M
if __name__ == '__main__':
    fire.Fire(main)