import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(cur_dir, '../..'))
sys.path.append(root_path)
import pickle

from src.model import NormLayer
import torch
import numpy as np


def test_norm_layer(x:np.ndarray, params:dict):
    d_model = x.shape[-1]
    norm = NormLayer(d_model)

    # Set the parameters
    # norm.g.data = torch.tensor(params['norm_layer/g:0'], dtype=torch.float32)
    # norm.b.data = torch.tensor(params['norm_layer/b:0'], dtype=torch.float32)

    # Load the parameters
    norm_params = norm.state_dict()
    mapping = {
        'norm_layer/g:0': 'g',
        'norm_layer/b:0': 'b'
    }
    for key in params.keys():
        if key in mapping:
            param_name = mapping[key]
            norm_params[param_name].data = torch.tensor(params[key], dtype=torch.float32)

    # Set the parameters back to the model
    norm.load_state_dict(norm_params)
    norm.eval()

    # Convert x to torch tensor
    x_tensor = torch.tensor(x, dtype=torch.float32)

    # Forward pass
    y_tensor = norm(x_tensor)
    y = y_tensor.detach().numpy()

    return y


if __name__ == "__main__":
    compare_path = '/Users/liyangliu/lyl.jpg/github/llm/gpt-2/debug/norm/'

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

    y = test_norm_layer(x, params)
    print('y:', y)

    # Compare with TensorFlow output
    ret = np.allclose(y, tf_y, atol=1e-5)
    print('ret:', ret)
    if ret:
        print("The outputs are close enough!")
    else:
        print("The outputs are not close enough!")
