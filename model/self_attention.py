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

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) 
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape   # x.shape = [batch, num_tokens, d_in]
        queries = self.W_query(x)       # query.shape = [batch, num_tokens, d_out]
        keys = self.W_key(x)            # key.shape = [batch, num_tokens, d_out]
        values = self.W_value(x)        # value.shape = [batch, num_tokens, d_out]

        attn_scores = queries @ keys.transpose(1, 2)
                                        # [batch, num_tokens, d_out] @ [batch, d_out, num_token]
                                        # attn_scores.shape = [batch, num_tokens, num_tokens]
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
                                        # [batch, num_tokens, num_tokens] @  [batch, num_tokens, d_out]
                                        # context_vec.shape = [batch, num_token, d_out]
        return context_vec
