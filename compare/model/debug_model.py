import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(cur_dir, '../..'))
sys.path.append(root_path)
import pickle

from src.model import Model
import torch
import numpy as np

# 设置打印精度为8位小数
torch.set_printoptions(precision=8)

def gpt2_compare(x:np.ndarray, params:dict, n_head, n_embd, n_layer, n_ctx, n_vocab, past=None):
    gpt2 = Model(n_embd, n_head, n_layer, n_ctx, n_vocab)

    # Load the parameters
    gpt2_params = gpt2.state_dict()
    print('gpt2_params:', gpt2_params.keys())
    mapping = {
        'model/wpe:0': 'wpe',
        'model/wte:0': 'wte',
        'model/h0/ln_1/g:0': 'models.0.ln_1.g',
        'model/h0/ln_1/b:0': 'models.0.ln_1.b',
        'model/h0/ln_2/g:0': 'models.0.ln_2.g',
        'model/h0/ln_2/b:0': 'models.0.ln_2.b',
        'model/h0/attn/c_attn/w:0': 'models.0.attn.c_attn.w',
        'model/h0/attn/c_attn/b:0': 'models.0.attn.c_attn.b',
        'model/h0/attn/c_proj/w:0': 'models.0.attn.c_proj.w',
        'model/h0/attn/c_proj/b:0': 'models.0.attn.c_proj.b',
        'model/h0/mlp/c_fc/w:0': 'models.0.mlp.c_fc.w',
        'model/h0/mlp/c_fc/b:0': 'models.0.mlp.c_fc.b',
        'model/h0/mlp/c_proj/w:0': 'models.0.mlp.c_proj.w',
        'model/h0/mlp/c_proj/b:0': 'models.0.mlp.c_proj.b',
        'model/ln_f/g:0': 'ln_f.g',
        'model/ln_f/b:0': 'ln_f.b',
    }
    for key in params.keys():
        if key in mapping:
            param_name = mapping[key]
            gpt2_params[param_name].data = torch.tensor(params[key], dtype=torch.float32)
            print(f'key: {key}, param_name: {param_name}')

    # Set the parameters back to the model
    gpt2.load_state_dict(gpt2_params)
    gpt2.eval()

    # Convert x to torch tensor
    x_tensor = torch.tensor(x, dtype=torch.int32)
    past_tensor = torch.tensor(past, dtype=torch.float32) if past is not None else None

    print('x_tensor:', x_tensor)
    print('past_tensor:', past_tensor)

    # Forward pass
    results = gpt2(x_tensor, past_tensor)
    y_tensor = results['logits']
    present_tensor = results['present']
    y = y_tensor.detach().numpy()
    present = present_tensor.detach().numpy()
    print('y_tensor:', y_tensor.shape)
    print('present_tensor:', present_tensor.shape)

    return y, present


if __name__ == "__main__":
    compare_path = '/Users/liyangliu/lyl.jpg/github/llm/gpt-2/debug/model/'

    # Load datas 
    with open(os.path.join(compare_path, "datas.pkl"), 'rb') as f:
        datas = pickle.load(f)

    # Example input
    x = datas['x']
    print('x:', x)

    # Example parameters
    params = datas['params']
    print('params:', params)

    tf_y = datas['y']
    print('tf_y:', tf_y)

    tf_present = datas['present']
    print('tf_present:', tf_present)

    n_head = datas['n_head']
    print('n_head:', n_head)
    n_embd = datas['n_embd']
    print('n_embd:', n_embd)
    n_layer = datas['n_layer']
    print('n_layer:', n_layer)
    n_ctx = datas['n_ctx']
    print('n_ctx:', n_ctx)
    n_vocab = datas['n_vocab']
    print('n_vocab:', n_vocab)

    past = datas['past']

    y, present = gpt2_compare(x, params, n_head, n_embd, n_layer, n_ctx, n_vocab, past)
    print('y:', y)
    print('present:', present)

    # Compare with TensorFlow output
    ret = np.allclose(y, tf_y, atol=1e-4)
    print('ret:', ret)
    if ret:
        print("The outputs y are close enough!")
    else:
        print("The outputs y are not close enough!")

    ret = np.allclose(present, tf_present, atol=1e-5)
    print('present ret:', ret)
    if ret:
        print("The outputs present are close enough!")
    else:
        print("The outputs present are not close enough!")
