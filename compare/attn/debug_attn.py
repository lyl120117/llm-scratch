import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(cur_dir, '../..'))
sys.path.append(root_path)
import pickle

from src.model import AttentionLayer
import torch
import numpy as np

# 设置打印精度为8位小数
torch.set_printoptions(precision=8)

def test_norm_layer(x:np.ndarray, params:dict, n_head, past=None):
    d_model = x.shape[-1]
    norm = AttentionLayer(d_model, n_head)

    # Load the parameters
    norm_params = norm.state_dict()
    print('norm_params:', norm_params.keys())
    mapping = {
        'attn_layer/c_attn/w:0': 'c_attn.w',
        'attn_layer/c_attn/b:0': 'c_attn.b',
        'attn_layer/c_proj/w:0': 'c_proj.w',
        'attn_layer/c_proj/b:0': 'c_proj.b',
    }
    for key in params.keys():
        if key in mapping:
            param_name = mapping[key]
            norm_params[param_name].data = torch.tensor(params[key], dtype=torch.float32)
            print(f'key: {key}, param_name: {param_name}')

    # Set the parameters back to the model
    norm.load_state_dict(norm_params)
    norm.eval()

    # Convert x to torch tensor
    x_tensor = torch.tensor(x, dtype=torch.float32)
    past_tensor = torch.tensor(past, dtype=torch.float32) if past is not None else None

    # Forward pass
    y_tensor, present_tensor = norm(x_tensor, past_tensor)
    y = y_tensor.detach().numpy()
    present = present_tensor.detach().numpy()
    print('y_tensor:', y_tensor.shape)
    print('present_tensor:', present_tensor.shape)

    return y, present


if __name__ == "__main__":
    compare_path = '/Users/liyangliu/lyl.jpg/github/llm/gpt-2/debug/attn/'

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

    past = datas['past']

    y, present = test_norm_layer(x, params, n_head, past)
    print('y:', y)
    print('present:', present)

    # Compare with TensorFlow output
    ret = np.allclose(y, tf_y, atol=1e-5)
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
