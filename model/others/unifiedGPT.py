import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import inspect

@dataclass
class iGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer_main: int = 8
    n_layer_idea: int = 4
    n_head: int = 8
    n_embd_main: int = 768
    n_embd_idea: int = 512
    idea_dim: int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            )
        )

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
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, add_idea=False):
        super().__init__()
        self.add_idea = add_idea

        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x, idea_vector=None):
        x = x + self.attn(self.ln_1(x))
        if self.add_idea and idea_vector is not None:
            B, T, C = x.shape
            idea_3d = idea_vector.unsqueeze(1).expand(-1, T, -1)
            x = x + self.mlp(self.ln_2(x + idea_3d))
        else:
            x = x + self.mlp(self.ln_2(x))
        return x

class iGPT(nn.Module):
    def __init__(self, config: iGPTConfig):
        super().__init__()
        self.config = config

        # iGPT sub-block
        self.wte_i = nn.Embedding(config.vocab_size, config.n_embd_idea)
        self.wpe_i = nn.Embedding(config.block_size, config.n_embd_idea)
        self.blocks_i = nn.ModuleList([
            Block(
                n_embd=config.n_embd_idea,
                n_head=config.n_head,
                block_size=config.block_size,
                add_idea=False
            )
            for _ in range(config.n_layer_idea)
        ])
        self.ln_f_i = nn.LayerNorm(config.n_embd_idea)
        self.idea_head = nn.Linear(config.n_embd_idea, config.idea_dim, bias=False)

        # main GPT sub-block
        self.wte_g = nn.Embedding(config.vocab_size, config.n_embd_main)
        self.wpe_g = nn.Embedding(config.block_size, config.n_embd_main)
        self.blocks_g = nn.ModuleList([
            Block(
                n_embd=config.n_embd_main,
                n_head=config.n_head,
                block_size=config.block_size,
                add_idea=True
            )
            for _ in range(config.n_layer_main)
        ])
        self.ln_f_g = nn.LayerNorm(config.n_embd_main)
        self.lm_head = nn.Linear(config.n_embd_main, config.vocab_size, bias=False)

        # optional weight sharing
        # comment out if you don't want wte_g to share with lm_head
        self.wte_g.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, ix, targets=None):
        """
        x:       (B, T_main) tokens for main GPT
        ix:      (B, T_idea) tokens for iGPT
        targets: (B, T_main) for cross-entropy
        """
        device = next(self.parameters()).device

        # iGPT sub-model
        ix = ix.to(device)
        B_i, T_i = ix.size()
        pos_i = torch.arange(T_i, device=device, dtype=torch.long)
        tok_i = self.wte_i(ix)
        pos_emb_i = self.wpe_i(pos_i)
        hidden_i = tok_i + pos_emb_i

        for block in self.blocks_i:
            hidden_i = block(hidden_i)

        hidden_i = self.ln_f_i(hidden_i)
        # last token's hidden => idea vector
        idea_vector = self.idea_head(hidden_i[:, -1, :])  # (B_i, idea_dim)

        # main GPT sub-model
        x = x.to(device)
        B_m, T_m = x.size()
        pos_m = torch.arange(T_m, device=device, dtype=torch.long)
        tok_m = self.wte_g(x)
        pos_emb_m = self.wpe_g(pos_m)
        hidden_m = tok_m + pos_emb_m

        for block in self.blocks_g:
            hidden_m = block(hidden_m, idea_vector=idea_vector)
        

        hidden_m = self.ln_f_g(hidden_m)
        logits = self.lm_head(hidden_m)  # (B_m, T_m, vocab_size)
    
        loss = None
        if targets is not None:
            targets = targets.to(device)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device, master_process=True):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for (n, p) in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for (n, p) in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, "
                  f"with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, "
                  f"with {num_nodecay_params:,} parameters")

        # check for fused AdamW
        import torch.optim
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer

# Example usage:
#
# config = iGPT
#Config(
#     block_size=256,
#     vocab_size=50304,
#     n_layer_main=8,
#     n_layer_idea=4,
#     n_head=12,
#     n_embd_main=768,
#     n_embd_idea=512,
#     idea_dim=768
# )
# model = iGPT
#(config).cuda()
# optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device="cuda", master_process=True)
#
# # Suppose x, ix, and y are your input indices for main GPT, iGPT, and targets:
# x, ix, y = get_some_batch()  # shapes: x=(B, T_main), ix=(B, T_idea), y=(B, T_main)
#
# logits, loss = model(x, ix, targets=y)
# loss.backward()
# optimizer.step()
