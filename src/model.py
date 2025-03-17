import torch
import math

class NormLayer(torch.nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(NormLayer, self).__init__()
        self.eps = eps
        self.g = torch.nn.Parameter(torch.ones(d_model)).to(torch.float32)
        self.b = torch.nn.Parameter(torch.zeros(d_model)).to(torch.float32)
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"
        assert x.shape[-1] == self.d_model, f"Expected last dimension of size {self.d_model}, got {x.shape[-1]}"
        assert x.dim() >= 2, "Input must have at least 2 dimensions"

        u = x.mean(dim=-1, keepdim=True)
        s = torch.square(x - u).mean(dim=-1, keepdim=True)
        std = 1 / torch.sqrt(s + self.eps)
        x = (x - u) * std
        return self.g * x + self.b


class Conv1d(torch.nn.Module):
    def __init__(self, nx, nf):
        super(Conv1d, self).__init__()
        self.nx = nx # d_model
        self.nf = nf # d_model * 3 or d_model
        self.w = torch.nn.Parameter(torch.randn(1, nx, nf)).to(torch.float32)
        self.b = torch.nn.Parameter(torch.zeros(nf)).to(torch.float32)

    def forward(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"
        assert x.shape[-1] == self.nx, f"Expected last dimension of size {self.nx}, got {x.shape[-1]}"
        assert x.dim() >= 2, "Input must have at least 2 dimensions"

        *start, _ = x.shape
        # print('Conv1d x:', x.shape, ', start:', start, ', nx:', self.nx, ', nf:', self.nf)
        c = torch.reshape(torch.matmul(torch.reshape(x, [-1, self.nx]), torch.reshape(self.w, [-1, self.nf])) + self.b, start + [self.nf])
        return c



class AttentionLayer(torch.nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = d_model // n_head

        self.c_attn = Conv1d(d_model, d_model * 3)
        self.c_proj = Conv1d(d_model, d_model)

    def split_states(self, x: torch.Tensor, n):
        """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
        *start, m = x.shape
        # print('split_states x:', x.shape)
        out = torch.reshape(x, start + [n, m // n])
        # print('split_states out:', out.shape)
        return out
    
    def merge_states(self, x: torch.Tensor):
        """Smash the last two dimensions of x into a single dimension."""
        *start, a, b = x.shape
        return torch.reshape(x, start + [a * b])

    def split_heads(self, x: torch.Tensor):
        # From [batch, sequence, features] to [batch, heads, sequence, features]

        assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"
        # print('split_heads x:', x.shape)

        c = self.split_states(x, self.n_head)
        # print('split_heads c:', c.shape)
        out = torch.transpose(c, 2, 1)
        # print('split_heads out:', out.shape)
        return out
    
    def merge_heads(self, x: torch.Tensor):
        # Reverse of split_heads
        return self.merge_states(torch.transpose(x, 2, 1))
    
    def attention_mask(self, nd, ns, *, dtype):
        i = torch.arange(nd)[:, None]
        j = torch.arange(ns)
        m = i >= j - ns + nd
        return m.to(dtype=dtype)

    def mask_attn_weights(self, w: torch.Tensor):
        _, _, nd, ns = w.shape
        b = self.attention_mask(nd, ns, dtype=w.dtype)
        b = b.reshape(1, 1, nd, ns)
        w = w * b - torch.tensor(1e10, dtype=w.dtype) * (1 - b)
        return w
    
    def softmax(self, x: torch.Tensor, dim=-1):
        m, _ = x.max(dim=dim, keepdim=True)
        x = x - m
        ex = torch.exp(x)
        return ex / torch.sum(ex, dim=dim, keepdim=True)
    
    def multihead_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # print('multihead_attn q:', q)
        # print('multihead_attn q:', q.shape)
        # print('multihead_attn k:', k)
        # print('multihead_attn k:', k.shape)
        # print('multihead_attn v:', v)
        # print('multihead_attn v:', v.shape)
        # q, k, v has shape [batch, heads, sequence, features]
        assert isinstance(q, torch.Tensor), "Input must be a torch.Tensor"
        assert isinstance(k, torch.Tensor), "Input must be a torch.Tensor"
        assert isinstance(v, torch.Tensor), "Input must be a torch.Tensor"
        assert k.shape == v.shape, f"k and v must have the same shape, got {k.shape} and {v.shape}"

        w = torch.matmul(q, k.transpose(-2, -1))
        w = w / (self.head_dim ** 0.5)

        w = self.mask_attn_weights(w)
        w = self.softmax(w)
        a = torch.matmul(w, v)
        return a

    def forward(self, x: torch.Tensor, past: torch.Tensor=None):
        assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"
        assert x.shape[-1] == self.d_model, f"Expected last dimension of size {self.d_model}, got {x.shape[-1]}"
        assert x.dim() >= 2, "Input must have at least 2 dimensions"
        # assert past is not None and isinstance(past, torch.Tensor), "past must be a torch.Tensor"

        c = self.c_attn(x)
        # print('forward c:', c)
        # print('forward c:', c.shape)
        split_size = int(c.shape[-1] / 3)
        q, k, v = map(self.split_heads, torch.split(c, split_size, dim=2))
        # print('forward q:', q)
        # print('forward q:', q.shape)
        # print('forward k:', k)
        # print('forward k:', k.shape)
        # print('forward v:', v)
        # print('forward v:', v.shape)
        present = torch.stack([k, v], dim=1)
        # print('forward present:', present.shape)

        if past is not None:
            # print('forward past:', past)
            # print('forward past:', past.shape)
            pk, pv = torch.unbind(past, dim=1)
            # print('forward pk:', pk.shape, ', k:', k.shape)
            k = torch.cat([pk, k], dim=-2)
            # print('forward cat k:', k.shape)
            v = torch.cat([pv, v], dim=-2)
        
        a = self.multihead_attn(q, k, v)
        # print('forward a:', a)
        # print('forward a:', a.shape)
        a = self.merge_heads(a)
        # print('forward merge_heads a:', a.shape)
        a = self.c_proj(a)
        # print('forward c_proj a:', a.shape)
        return a, present


class MLPLayer(torch.nn.Module):
    def __init__(self, nx, n_state:int):
        super(MLPLayer, self).__init__()
        self.c_fc = Conv1d(nx, n_state)
        self.c_proj = Conv1d(n_state, nx)

    def gelu(self, x: torch.Tensor):
        # 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))
        return 0.5 * x * (1 + torch.tanh((2 / math.pi) ** 0.5 * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"
        # print('MLPLayer c_fc:', self.c_fc.w.shape)
        # print('MLPLayer c_fc:', self.c_fc.b.shape)
        h = self.gelu(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

class AttentionBlock(torch.nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.ln_1 = NormLayer(d_model)
        self.ln_2 = NormLayer(d_model)
        self.attn = AttentionLayer(d_model, n_head, dropout)
        self.mlp = MLPLayer(d_model, d_model * 4)

    def forward(self, x: torch.Tensor, past: torch.Tensor=None):
        assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"
        # assert past is not None and isinstance(past, torch.Tensor), "past must be a torch.Tensor"
        a, present = self.attn(self.ln_1(x), past)
        # print('AttentionBlock x:', x.shape)
        # print('AttentionBlock a:', a.shape)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class Model(torch.nn.Module):
    def __init__(self, d_model, n_head, n_layer, n_context, vocab_size, dropout=0.1):
        super(Model, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.vocab_size = vocab_size

        self.wte = torch.nn.Parameter(torch.randn(vocab_size, d_model)).to(torch.float32)
        self.wpe = torch.nn.Parameter(torch.randn(n_context, d_model)).to(torch.float32)
        self.models = torch.nn.ModuleList([AttentionBlock(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = NormLayer(d_model)

    def expand_tile(self, value, size):
        """Add a new axis of given size."""
        if isinstance(value, torch.Tensor):
            value = value.to(torch.int64)
        else:
            value = torch.tensor(value, dtype=torch.int64)
        ndims = len(value.shape)
        return torch.tile(torch.unsqueeze(value, dim=0), [size] + [1] * ndims)

    def positions_for(self, tokens: torch.Tensor, past_length):
        batch_size, nsteps = tokens.shape
        return self.expand_tile(torch.arange(nsteps, dtype=tokens.dtype) + past_length, batch_size).to(tokens.dtype)

    def forward(self, x: torch.Tensor, past: torch.Tensor=None):
        assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"
        assert x.dtype == torch.int32, "Input must be of type int32"
        assert x.dim() == 2, "Input must have 2 dimensions"
        # assert past is not None and isinstance(past, torch.Tensor), "past must be a torch.Tensor"
        batch, sequence = x.shape
        # print('Model x:', x)
        past_length = past.shape[-2] if past is not None else 0
        # print('### Model past_length:', past_length)

        positions = self.positions_for(x, past_length)
        # print('### Model positions:', positions.shape)

        tX = self.wte[x]
        # print('### Model tX:', tX.shape)
        pX = self.wpe[positions]
        # print('### Model pX:', pX.shape)
        
        h = tX + pX
        # print('### Model h:', h.shape)

        # Transformer
        presents = []
        # print('### Model past:', past.shape)
        pasts = torch.unbind(past, dim=1) if past is not None else [None] * self.n_layer
        # print('### Model pasts:', len(pasts))

        assert len(pasts) == self.n_layer, f"Expected {self.n_layer} pasts, got {len(pasts)}"

        for model, p in zip(self.models, pasts):
            h, present = model(h, p)
            presents.append(present)
        # print('### Model presents:', len(presents))
        present = torch.stack(presents, dim=1)
        results = {}
        results['present'] = present

        h = self.ln_f(h)
        # print('### Model h:', h.shape)

        # Language model loss.  Do tokens <n predict token n?
        h_half = torch.reshape(h, [batch * sequence, self.d_model])
        # print('### Model h_half:', h_half.shape)
        # print('### Model wte:', self.wte.shape)
        logits = torch.matmul(h_half, torch.transpose(self.wte, 0, 1))
        # print('### Model logits:', logits.shape)
        logits = torch.reshape(logits, [batch, sequence, self.vocab_size])
        # print('### Model logits:', logits.shape)
        results['logits'] = logits
        return results

def get_model(name:str):
    hparams = {
        '124M': {
            'n_embd': 768,
            'n_head': 12,
            'n_layer': 12,
            'n_ctx': 1024,
            'n_vocab': 50257
        },
        '355M': {
            'n_embd': 1024,
            'n_head': 16,
            'n_layer': 24,
            'n_ctx': 1024,
            'n_vocab': 50257
        },
        '774M': {
            'n_embd': 1280,
            'n_head': 20,
            'n_layer': 36,
            'n_ctx': 1024,
            'n_vocab': 50257
        },
        '1558M': {
            'n_embd': 1600,
            'n_head': 25,
            'n_layer': 48,
            'n_ctx': 1024,
            'n_vocab': 50257
        },
    }
    assert name in hparams, f"Model {name} not found, available models are: {list(hparams.keys())}"
    params = hparams[name]
    n_embd = params['n_embd']
    n_head = params['n_head']
    n_layer = params['n_layer']
    n_ctx = params['n_ctx']
    n_vocab = params['n_vocab']
    model = Model(n_embd, n_head, n_layer, n_ctx, n_vocab)
    return model