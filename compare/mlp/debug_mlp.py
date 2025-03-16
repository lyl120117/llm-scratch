import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(cur_dir, '../..'))
sys.path.append(root_path)
import pickle

from src.model import MLPLayer
import torch
import numpy as np

# 设置打印精度为8位小数
torch.set_printoptions(precision=8)

def test_norm_layer(x:np.ndarray, params:dict):
    nx = x.shape[-1]
    mlp = MLPLayer(nx, nx * 4)

    # Load the parameters
    mlp_params = mlp.state_dict()
    print('norm_params:', mlp_params.keys())
    mapping = {
        'mlp/c_fc/w:0': 'c_fc.w',
        'mlp/c_fc/b:0': 'c_fc.b',
        'mlp/c_proj/w:0': 'c_proj.w',
        'mlp/c_proj/b:0': 'c_proj.b',
    }
    for key in params.keys():
        if key in mapping:
            param_name = mapping[key]
            mlp_params[param_name].data = torch.tensor(params[key], dtype=torch.float32)
            print(f'key: {key}, param_name: {param_name}')

    # Set the parameters back to the model
    mlp.load_state_dict(mlp_params)
    mlp.eval()

    # Convert x to torch tensor
    x_tensor = torch.tensor(x, dtype=torch.float32)

    # Forward pass
    y_tensor = mlp(x_tensor)
    y = y_tensor.detach().numpy()
    print('y_tensor:', y_tensor.shape)

    return y


if __name__ == "__main__":
    compare_path = '/Users/liyangliu/lyl.jpg/github/llm/gpt-2/debug/mlp/'

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

    n_head = datas['n_head']
    print('n_head:', n_head)

    y = test_norm_layer(x, params)
    print('y:', y)

    # Compare with TensorFlow output
    ret = np.allclose(y, tf_y, atol=1e-5)
    print('ret:', ret)
    if ret:
        print("The outputs are close enough!")
    else:
        print("The outputs are not close enough!")
