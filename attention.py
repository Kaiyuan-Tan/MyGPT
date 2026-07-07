import torch.nn as nn
import torch

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.W_query  # [x, d_out]
        keys = x @ self.W_key       # [x, d_out]
        values = x @ self.W_value   # [x, d_out]

        attn_score = queries @ keys.T   # [x, x]
        attn_weight = torch.softmax(attn_score / keys.shape[-1]**0.5, dim=-1)
        # [x, x]
        context_vec = attn_weight @ values
        # [x, d_out]
        return context_vec

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)  # [x, d_out]
        keys = self.W_key(x)       # [x, d_out]
        values = self.W_value(x)   # [x, d_out]

        attn_score = queries @ keys.T   # [x, x]
        attn_weight = torch.softmax(attn_score / keys.shape[-1]**0.5, dim=-1)
        # [x, x]
        context_vec = attn_weight @ values
        # [x, d_out]
        return context_vec
    
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],# Your    (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one     6 (x^5)
    [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in = inputs.shape[1] #B
d_out = 2 #C

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)

# Query & Key
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
# attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)

# mask & renormalize
context_length = attn_scores.shape[0]

# mask_simple = torch.tril(torch.ones(context_length, context_length))
# masked_simple = attn_weights*mask_simple
# row_sums = masked_simple.sum(dim=1, keepdim=True)
# masked_simple_norm = masked_simple / row_sums
# print(masked_simple_norm)

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1) # masked attention weight

