import torch
import torch.nn as nn
from model.layernorm import LayerNorm
from model.mha import MultiHeadAttention
from model.GELU import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.Attention = MultiHeadAttention(
            d_in = cfg["emb_dim"], 
            d_out = cfg["emb_dim"], 
            context_length = cfg["context_length"], 
            dropout = cfg["drop_rate"], 
            num_heads = cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"])
        self.Layernorm1 = LayerNorm(cfg["emb_dim"])
        self.Layernorm2 = LayerNorm(cfg["emb_dim"])
        self.ffd = FeedForward(cfg)
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        residual = x
        x = self.Layernorm1(x)
        x = self.Attention(x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.Layernorm2(x)
        x = self.ffd(x)
        x = self.dropout(x)
        x = x + residual

        return x