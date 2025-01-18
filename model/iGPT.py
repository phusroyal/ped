"""
This is iGPT v0.1.0
- Encoder is sentence-transformers/all-mpnet-base-v2 (dim 768) (https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
- Decoder is GPT (from scratch)
- ivec is treated as a prepended token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from dataclasses import dataclass
from torch.optim.lr_scheduler import CosineAnnealingLR
from sentence_transformers import SentenceTransformer
from lightning.pytorch.utilities import grad_norm

@dataclass
class iGPTConfig:
    block_size: int = 256
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
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
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    # removed the 'idea_vector' argument / logic
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class iGPT(nn.Module):
    """
    GPT model that prepends a single 'idea token' derived from
    all-mpnet-base-v2 embeddings at the start of the sequence.
    """

    def __init__(self, config: iGPTConfig, st_model_name='sentence-transformers/all-mpnet-base-v2'):
        super().__init__()
        self.config = config

        # Load the Sentence Transformer model
        self.sbert = SentenceTransformer(st_model_name)
        # The output dimension is typically 768
        self.sbert_dim = 768

        # Project from sbert_dim -> GPT embedding dimension if needed
        if self.sbert_dim != config.n_embd:
            self.idea_proj = nn.Linear(self.sbert_dim, config.n_embd)
        else:
            self.idea_proj = nn.Identity()

        # GPT sub-block
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size+1, config.n_embd)
        self.blocks = nn.ModuleList([
            Block(
                n_embd=config.n_embd,
                n_head=config.n_head,
                block_size=config.block_size+1  # +1 for the prepended idea token
            )
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, sentence_list):
        """
        x: (B, T) GPT tokens
        sentence_list: List[str] of length B

        Returns: (B, T, vocab_size) for language model tokens (excluding the idea token)
        """
        B, T = x.size()

        # Get the idea embeddings from all-mpnet-base-v2
        with torch.no_grad():
            idea_vecs = self.sbert.encode(sentence_list, convert_to_tensor=True, device=x.device)  # shape (B, 768)
        # Project if needed
        idea_emb = self.idea_proj(idea_vecs)  # shape (B, n_embd)
        idea_emb = idea_emb.unsqueeze(1)      # shape (B,1,n_embd)

        # GPT embeddings
        tok_emb = self.wte(x)  # (B, T, n_embd)
        hidden = torch.cat([idea_emb, tok_emb], dim=1)  # (B, T+1, n_embd)
        pos_ids = torch.arange(0, T+1, device=x.device)  # length T+1
        pos_emb = self.wpe(pos_ids).unsqueeze(0).expand(B, T+1, -1)
        hidden = hidden + pos_emb  # (B, T+1, n_embd)

        # Forward through GPT
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.ln_f(hidden)  # (B, T+1, n_embd)

        # skip the idea token when producing logits for next-word prediction
        logits = self.lm_head(hidden[:, 1:, :])  # shape (B, T, vocab_size)
        return logits

class NotMyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.config = iGPTConfig()
        self.network = iGPT(self.config)
        self.lr = 6e-4
        self.weight_decay = 0.1 


    def training_step(self, batch, batch_idx):
        self.log('global_step', self.global_step)

        x, y, sentence_list = batch
        predicted_y = self.network(x, sentence_list)
        loss = F.cross_entropy(predicted_y.view(-1, predicted_y.size(-1)), y.view(-1))
        # Logging
        self.log("ahihi/Train Loss", loss, prog_bar=True)
        return loss
        

    def validation_step(self, batch, batch_idx):
        x, y, sentence_list = batch
        predicted_y = self.network(x, sentence_list)
        loss = F.cross_entropy(predicted_y.view(-1, predicted_y.size(-1)), y.view(-1))

        self.log("ahihi/Val loss", loss, prog_bar=True)
        return loss
        

    def test_step(self, batch, batch_idx):
        pass


    def forward(self, batch):
        x, ix = batch
        logits = self.network(x, ix)
        return logits
    
    
    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for (n, p) in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for (n, p) in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            # fused=use_fused
        )

        lr_scheduler = {'scheduler': CosineAnnealingLR(optimizer=optimizer, T_max=500, eta_min=self.lr*0.1),
                    'name': 'learning_rate',
                    'interval':'step',
                    'frequency': 1}
        
        return [optimizer], [lr_scheduler]

    def on_before_optimizer_step(self, optimizer):
        norm = grad_norm(self.network, norm_type=2)
        avg_norm = sum(norm.values())/len(norm)
        self.log('ahihi/norm', avg_norm, prog_bar=True)

            