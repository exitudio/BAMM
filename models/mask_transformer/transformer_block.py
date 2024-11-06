import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat

class Attention(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x, mask):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            att[mask] = float('-inf')
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1, dim_feedforward=1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks
        self.n_head = self.blocks[0].attn.n_head
        # self.apply(self._init_weights)

    def forward(self, x, mask=None, word_emb=None, word_att_mask=None):
        x = x.permute(1, 0, 2) # T b e -> b T e
        if mask is not None:
            if len(mask.shape) == 2:
                mask = repeat(mask, 'T1 T2 -> b h T1 T2', 
                              b=x.shape[0], h=self.n_head)
        for block in self.blocks:
            if isinstance(block, TransformerDecoderLayer):
                x = block(x, word_emb, word_att_mask)
            else:
                x = block(x, mask)
        x = x.permute(1, 0, 2) # b T e -> T b e
        return x
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        
class CrossAttention(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(block_size, 77)).view(1, 1, block_size, 77))
        self.n_head = n_head

    def forward(self, x,word_emb, mask):
        B, T, C = x.size()
        B, N, D = word_emb.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(word_emb).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(word_emb).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, N) -> (B, nh, T, N)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            att[mask] = float('-inf')
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, N) x (B, nh, N, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1, dim_feedforward=1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.attn = CrossAttention(embed_dim, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x,word_emb, mask=None):
        x = x + self.attn(self.ln1(x), self.ln3(word_emb), mask)
        x = x + self.mlp(self.ln2(x))
        return x

    


        

