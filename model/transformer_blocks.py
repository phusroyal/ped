import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)                   # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, mlp_ratio=4.0):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, mlp_ratio * n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(mlp_ratio * n_embd, n_embd)
        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    # removed the 'idea_vector' argument / logic
    def __init__(self, n_embd, n_head, mlp_ratio=4.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.ln_1 = norm_layer(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = norm_layer(n_embd)
        self.mlp = MLP(n_embd, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x