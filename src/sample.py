from src.model import get_model
from src.encoder import get_encoder
import torch
import os
import numpy as np

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        # get the top k logits
        values, _ = torch.topk(logits, k=k)
        min_values = values[:, -1].unsqueeze(-1)
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
    
    return _top_k()

def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, _ = logits.shape
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    indices = torch.stack(
        torch.range(0, batch),
        # Number of indices to include
        torch.maximum(torch.sum(cumulative_probs < p, dim=-1) - 1, torch.tensor(0)),
    )
    min_values = sorted_logits[indices, -1].unsqueeze(-1)
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

def sample_sequence(
    name,
    model_path,
    n_vocab,
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

    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist.")
    # Load the model parameters
    model_params = torch.load(model_path)
    model.load_state_dict(model_params)

    # Move the model to the specified device
    model.to(device)

    model.eval()
    print("Model loaded successfully.")

    # Load the encoder
    encoder = get_encoder(os.path.dirname(model_path))
    if encoder is None:
        raise ValueError(f"Encoder not found for model {name}.")
    print("Encoder loaded successfully.")

    raw_text = "Hello, how are you?"
    context_tokens = encoder.encode(raw_text)
    print("Context tokens:", context_tokens)

    # x = np.random.randint(0, n_vocab, (batch_size, length), dtype=np.int32)
    # Sample sequences
    with torch.no_grad():
        x_tensor = torch.tensor([context_tokens], dtype=torch.int32).to(device)
        past = None
        results = model(x_tensor, past)
        logits = results['logits']
        logits = logits[:, -1, :] / temperature
        # logits = top_k_logits(logits, top_k)
        # print('top_k_logits logits:', logits.shape)
        # logits = top_p_logits(logits, top_p)
        # print('top_p_logits logits:', logits.shape)
        present_tensor = results['present']
        # y_tensor = y_tensor.squeeze(1)
        logits = torch.nn.functional.softmax(logits, dim=-1)
        print('logits:', logits.shape)
        # y_tensor = torch.argmax(y_tensor, dim=-1)
        y_tensor = torch.multinomial(logits, num_samples=1)
        # present_tensor = present_tensor.squeeze(1)

        # Convert tensors to numpy arrays
        y = y_tensor.detach().cpu().numpy()
        present = present_tensor.detach().cpu().numpy()
        print('raw_text:', raw_text)
        print('context_tokens:', context_tokens)
        print('y:', y)
        text = encoder.decode(y[0])
        print('Decoded text:', text)
        print('present:', present.shape)


if __name__ == "__main__":
    # Example usage
    # name = '124M'
    name = '1558M'
    model_path = f'weights/{name}.pt'
    n_vocab = 50257
    length = 1
    start_token = 0
    batch_size = 1
    temperature = 1.0
    top_k = 1
    top_p = 0.95
    device = 'cpu'

    sample_sequence(name, model_path, n_vocab, length, start_token, batch_size, temperature, top_k, top_p, device)