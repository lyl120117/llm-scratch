import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(cur_dir, '..'))
sys.path.append(root_path)

from src.model import get_model
import torch
import os
import numpy as np


def sample_sequence(
    name,
    length,
    start_token,
    batch_size=1,
    temperature=1.0,
    top_k=40,
    top_p=0.95,
    device='cpu',
):
    """
    Sample a sequence of tokens from the model.
    
    Args:
        model: The model to sample from.
        length: The length of the sequence to sample.
        start_token: The token to start the sequence with.
        batch_size: The number of sequences to sample.
        temperature: The temperature for sampling.
        top_k: The top-k sampling parameter.
        top_p: The top-p sampling parameter.
        device: The device to use for computation.

    Returns:
        A tensor containing the sampled sequences.
    """
    # Load the model
    model = get_model(name)

    model_dir = os.path.join('/Users/liyangliu/lyl.jpg/github/llm/gpt-2/outputs', name)
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory {model_dir} does not exist.")
    
    # Load the model parameters
    model_params = model.state_dict()
    for key, param in model_params.items():
        param_file = os.path.join(model_dir, f"{key}.npy")
        if os.path.exists(param_file):
            param_data = np.load(param_file)
            param.data = torch.tensor(param_data, dtype=torch.float32)
            print(f'Loaded parameter from {param_file}, shape: {param_data.shape}')
        else:
            print(f'Parameter file {param_file} does not exist.')

    # Load the model weights
    model.load_state_dict(model_params)
    # Move the model to the specified device
    model.to(device)

    model.eval()
    print("Model loaded successfully.")

    # Save the model parameters to a pt file
    model_path = os.path.join(model_dir, '..', f'{name}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model parameters saved to {model_path}")


if __name__ == "__main__":
    # Example usage
    # name = '124M'
    name = '1558M'
    length = 1
    start_token = 0
    batch_size = 1
    temperature = 1.0
    top_k = 1
    top_p = 0.95
    device = 'cpu'

    sample_sequence(name, length, start_token, batch_size, temperature, top_k, top_p, device)