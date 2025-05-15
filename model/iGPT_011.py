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

from model.transformer_blocks import Block, CausalSelfAttention, MLP

@dataclass
class iGPTConfig:
    id_pred_block_size: int = 32
    id_pred_n_layer: int = 6
    id_pred_n_head: int = 12
    id_pred_n_embd: int = 384
    id_dec_block_size: int = 32
    id_dec_n_layer: int = 6
    id_dec_n_head: int = 12
    id_dec_n_embd: int = 384
    mlp_ratio: float = 4.0
    vocab_size: int = 50304


class PretrainedIdearEncoder(nn.Module):
    def __init__(self, st_model_name='sentence-transformers/all-mpnet-base-v2'):
        super().__init__()
        # Load the Sentence Transformer model
        self.sbert = SentenceTransformer(st_model_name)
        # The output dimension is typically 768
        self.sbert_dim = 768

    def forward(self, sentence_list):
        # Get the idea embeddings from all-mpnet-base-v2
        with torch.no_grad():
            idea_vecs = self.sbert.encode(sentence_list, convert_to_tensor=True)  # shape (B, 768)
        return idea_vecs
    
class IdeaPredictor(nn.Module):
    def __init__(self, 
                block_size=32,
                input_dim=768,
                predictor_embed_dim=384,
                predictor_depth=12,
                predictor_num_heads=12,
                mlp_ratio=4.0,
                # qkv_bias=False,
                # qk_scale=None,
                # drop_rate=0.0,
                # attn_drop_rate=0.0,
                # drop_path_rate=0.0,
                # init_std=0.02,
                norm_layer=nn.LayerNorm,
                ):
        
        super().__init__()
        self.block_size = block_size
        self.input_dim = input_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.predictor_depth = predictor_depth
        self.predictor_num_heads = predictor_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer

        # Project from idea encoder dimension to predictor dimension
        if predictor_embed_dim != input_dim:
            self.idea_proj = nn.Linear(input_dim, predictor_embed_dim, bias=False)            
            self.predictor_proj = nn.Linear(predictor_embed_dim, input_dim, bias=False)
            self.predictor_proj.SCALE_INIT = 1
            self.idea_proj.SCALE_INIT = 1
        else:
            self.idea_proj = nn.Identity()
            self.predictor_proj = nn.Identity()

        self.predictor_wpe = nn.Embedding(block_size+1, predictor_embed_dim)
        self.predictor_blocks = nn.ModuleList([
            Block(
                n_embd=predictor_embed_dim,
                n_head=predictor_num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                # qkv_bias=qkv_bias,
                # qk_scale=qk_scale,
                # drop_rate=drop_rate,
                # attn_drop_rate=attn_drop_rate,
                # drop_path_rate=drop_path_rate,
            )
            for _ in range(predictor_depth)
        ])

        self.predictor_ln = nn.LayerNorm(predictor_embed_dim)

        # might need weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.predictor_depth) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idea_vecs, target=None):
        """
        idea_vecs: (B, I, inputdim) from all-mpnet-base-v2
        Returns: (B, I, inputdim) 
        """
        B, I = idea_vecs.size(0), idea_vecs.size(1)

        # Project the idea embeddings
        idea_emb = self.idea_proj(idea_vecs)  # shape (B, I, predictor_embed_dim)

        # Add positional embeddings
        pos_ids = torch.arange(0, I, device=idea_vecs.device)  # length T
        pos_emb = self.predictor_wpe(pos_ids)  # shape  (B, I, predictor_embed_dim)
        hidden = idea_emb + pos_emb  # (B, I, predictor_embed_dim)

        # Forward through predictor blocks
        for block in self.predictor_blocks:
            hidden = block(hidden)
        hidden = self.predictor_ln(hidden)

        # Project back to the original dimension
        hidden = self.predictor_proj(hidden)
        return hidden

class IdeaDecoder(nn.Module):
    def __init__(self,
                block_size=32,
                vocab_size=50304,
                input_dim=768,
                decoder_embed_dim=384,
                decoder_depth=6,
                decoder_num_heads=12,
                mlp_ratio=4.0,
                # qkv_bias=False,
                # qk_scale=None,
                # drop_rate=0.0,
                # attn_drop_rate=0.0,
                # drop_path_rate=0.0,
                # init_std=0.02,
                norm_layer=nn.LayerNorm,
                ):
        super().__init__()
        self.block_size = block_size
        self.input_dim = input_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.vocab_size = vocab_size

        # Project from idea encoder dimension to predictor dimension
        if decoder_embed_dim != input_dim:
            self.idea_proj = nn.Linear(input_dim, decoder_embed_dim, bias=False)            
            self.decoder_proj = nn.Linear(decoder_embed_dim, input_dim, bias=False)
            self.decoder_proj.SCALE_INIT = 1
            self.idea_proj.SCALE_INIT = 1
        else:
            self.idea_proj = nn.Identity()
            self.decoder_proj = nn.Identity()

        self.decoder_wte = nn.Embedding(vocab_size, decoder_embed_dim)
        self.decoder_wpe = nn.Embedding(block_size+1, decoder_embed_dim)
        self.block = nn.ModuleList([
            Block(
                n_embd=decoder_embed_dim,
                n_head=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                # qkv_bias=qkv_bias,
                # qk_scale=qk_scale,
                # drop_rate=drop_rate,
                # attn_drop_rate=attn_drop_rate,
                # drop_path_rate=drop_path_rate,
            )
            for _ in range(decoder_depth)
        ])

        self.decoder_ln = nn.LayerNorm(decoder_embed_dim)
        self.decoder_lm_head = nn.Linear(decoder_embed_dim, vocab_size, bias=False)
        # Weight tying
        self.decoder_wte.weight = self.decoder_lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.decoder_depth) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, idea_vec, target=None):
        """
        x: (B*I, T) GPT tokens 
        idea_vecs: (B*I, inputdim) from all-mpnet-base-v2

        Returns: 
        yt: (B, I, T, vocab_size) logits for next-word prediction
        ys: (B, I, predictor_embed_dim) for next-idea prediction
        """
        B, T = x.size()   # now B and T are ints

        # Project the idea embeddings
        idea_vec = self.idea_proj(idea_vec)

        # forward the token and positional embeddings
        token_emb = self.decoder_wte(x)  # (B*I, T, decoder_embed_dim)
        pos_ids = torch.arange(0, T, device=x.device)
        pos_emb = self.decoder_wpe(pos_ids)       
        idea_token = idea_vec.unsqueeze(1)                 # (B*I, 1, D)
        token_and_pos = token_emb + pos_emb               # (B*I, T, D)
        hidden = torch.cat([idea_token, token_and_pos], 1) # (B*I, T+1, D)


        # Forward through decoder blocks
        for block in self.block:
            hidden = block(hidden)
        hidden = self.decoder_ln(hidden)
        # skip the idea token when producing logits for next-word prediction
        logits = self.decoder_lm_head(hidden[:, 1:, :])  # shape (B*I, T, vocab_size)
        return logits


class iGPT(nn.Module):
    """
    GPT model that prepends a single 'idea token' derived from
    all-mpnet-base-v2 embeddings at the start of the sequence.
    """

    def __init__(self, config: iGPTConfig, st_model_name='sentence-transformers/all-mpnet-base-v2'):
        super().__init__()
        self.config = config

        # -- idea encoder
        # Load the Sentence Transformer model
        self.sbert = SentenceTransformer(st_model_name)
        self.sbert.eval()  # Set to evaluation mode
        self.sbert_dim = self.sbert.get_sentence_embedding_dimension()  # typically 768
        
        # -- idea predictor
        self.idea_predictor = IdeaPredictor(
            idea_size=config.id_pred_block_size,
            embed_dim=self.sbert_dim,
            predictor_embed_dim=config.id_pred_n_embd,
            predictor_depth=config.id_pred_n_layer,
            predictor_num_heads=config.id_pred_n_head,
            mlp_ratio=config.mlp_ratio,
            norm_layer=nn.LayerNorm
        )

        # -- idea decoder
        self.idea_decoder = IdeaDecoder(
            block_size=config.id_dec_block_size,
            vocab_size=config.vocab_size,
            input_dim=self.sbert_dim,
            decoder_embed_dim=config.id_dec_n_embd,
            decoder_depth=config.id_dec_n_layer,
            decoder_num_heads=config.id_dec_n_head,
            mlp_ratio=config.mlp_ratio,
            norm_layer=nn.LayerNorm
        )

    def forward(self, xt, xs):
        """
        xt: (B*I, T) GPT tokens 
        xs: (B, I) idea sentences

        Returns: 
        yt: (B, I, T, vocab_size) logits for next-word prediction
        ys: (B, I, predictor_embed_dim) for next-idea prediction
        """
        B, I, T = xt.size(), xt.size(1), xt.size(2)

        # Get the idea embeddings from all-mpnet-base-v2 for all sentences I in the batch
        with torch.no_grad():
            idea_vecs = self.sbert.encode(xs, convert_to_tensor=True, device=xt.device)  # shape (B, I, 768)
        
        # Pass to idea predictor
        ys = self.idea_predictor(idea_vecs) # shape (B, I, predictor_embed_dim)
        # flatten to (B*I, predictor_embed_dim)
        ys = ys.view(B*I, -1)

        # Pass to idea decoder
        yt = self.idea_decoder(xt, ys)

        return yt, ys, idea_vecs