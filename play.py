from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass
from lightning.pytorch.utilities import grad_norm

# -------------------------------------------
# GPT Building Blocks (same as your code but no iGPT logic)
# -------------------------------------------

class CausalSelfAttention(nn.Module):
    # same code as your existing GPT
    ...

class MLP(nn.Module):
    # same code as your existing GPT
    ...

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

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 768

class GPTWithSentenceTransformer(nn.Module):
    """
    GPT model that prepends a single 'idea token' derived from
    all-mpnet-base-v2 embeddings at the start of the sequence.
    """

    def __init__(self, config: GPTConfig, st_model_name='sentence-transformers/all-mpnet-base-v2'):
        super().__init__()
        self.config = config

        # 1) Load the Sentence Transformer model
        #    Usually you keep it in inference (no grad). But you can also set trainable if you want.
        self.sbert = SentenceTransformer(st_model_name)
        # The output dimension is typically 768
        self.sbert_dim = 768

        # 2) Project from sbert_dim -> GPT embedding dimension if needed
        # If your GPT dimension is also 768, you might not need this.
        # But let's be safe if the dimensions differ:
        if self.sbert_dim != config.n_embd:
            self.idea_proj = nn.Linear(self.sbert_dim, config.n_embd)
        else:
            self.idea_proj = nn.Identity()

        # 3) GPT sub-block
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
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, sentence_list):
        """
        x: (B, T) GPT tokens
        sentence_list: List of strings, length B.  # The text you want to encode with SBERT
                      Or you might pass these embeddings from a prior step if you want.
        """
        B, T = x.shape

        # 1) Get the idea embeddings from all-mpnet-base-v2
        #    This is typically done in float32 or float16.
        #    If you're on GPU, ensure your SBERT model is also on GPU.
        #    For example:
        #    self.sbert.to(x.device)
        #    embeddings = self.sbert.encode(...) might not be differentiable
        #    because sentence-transformers uses some internal logic. Usually we freeze it anyway.
        # Alternatively, do a manual approach with a huggingface model (see next section).
        with torch.no_grad():
            idea_vecs = self.sbert.encode(sentence_list, convert_to_tensor=True, device=x.device)  # shape (B, 768)
        # Project if needed
        idea_emb = self.idea_proj(idea_vecs)  # shape (B, n_embd)
        idea_emb = idea_emb.unsqueeze(1)      # shape (B,1,n_embd)

        # 2) GPT token embeddings
        pos_ids = torch.arange(T, dtype=torch.long, device=x.device)
        tok_emb = self.wte(x)  # (B, T, n_embd)
        pos_emb = self.wpe(pos_ids)  # (T, n_embd)
        pos_emb = pos_emb.unsqueeze(0).expand(B, T, -1)
        hidden = tok_emb + pos_emb

        # We must also create position embeddings for the extra token. Let's define it as position=0
        idea_pos_emb = self.wpe(torch.tensor([0], device=x.device)).view(1,1,-1)
        idea_token = idea_emb + idea_pos_emb  # (B,1,n_embd)

        # Shift the original positions by +1
        shift_pos_ids = torch.arange(1, T+1, device=x.device)
        shift_pos_emb = self.wpe(shift_pos_ids).unsqueeze(0).expand(B, T, -1)
        hidden = tok_emb + shift_pos_emb

        # Prepend the idea token
        hidden = torch.cat([idea_token, hidden], dim=1)  # (B, T+1, n_embd)

        # 3) Forward through GPT
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.ln_f(hidden)  # (B, T+1, n_embd)

        # We skip the idea token when we produce logits for next-word prediction
        logits = self.lm_head(hidden[:, 1:, :])  # shape (B, T, vocab_size)
        return logits

# -----------------------------------------------------------
# A LightningModule using GPTWithSentenceTransformer
# -----------------------------------------------------------
class NotMyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.config = GPTConfig(
            block_size=256,
            vocab_size=50304,
            n_layer=12,
            n_head=8,
            n_embd=768
        )
        self.network = GPTWithSentenceTransformer(self.config)
        self.lr = 6e-4
        self.weight_decay = 0.1

    def training_step(self, batch, batch_idx):
        """
        Suppose batch = (x, y, sentence_list).
        - x, y: GPT tokens & targets shape (B, T)
        - sentence_list: list of B strings
        """
        x, y, sentence_list = batch
        predicted_y = self.network(x, sentence_list)
        loss = F.cross_entropy(predicted_y.view(-1, predicted_y.size(-1)), y.view(-1))
        self.log("Train/Loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, sentence_list = batch
        predicted_y = self.network(x, sentence_list)
        loss = F.cross_entropy(predicted_y.view(-1, predicted_y.size(-1)), y.view(-1))
        self.log("Val/Loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for (n, p) in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for (n, p) in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, betas=(0.9, 0.95), eps=1e-8)
        lr_scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=500, eta_min=self.lr*0.1),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]

    def on_before_optimizer_step(self, optimizer):
        norm = grad_norm(self.network, norm_type=2)
        avg_norm = sum(norm.values())/len(norm)
        self.log('grad_norm', avg_norm, prog_bar=True)
