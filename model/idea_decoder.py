import torch.nn.functional as F
from torch import nn
from model.transformer_blocks import Block
import torch


class IdeaDecoder(nn.Module):
    def __init__(self,
                block_size=32,
                vocab_size=50304,
                input_dim=768,
                decoder_embed_dim=384,
                decoder_depth=6,
                decoder_num_heads=12,
                mlp_ratio=4,
                # qkv_bias=False,
                # qk_scale=None,
                # drop_rate=0.0,
                # attn_drop_rate=0.0,
                # drop_path_rate=0.0,
                # init_std=0.02,
                norm_layer=nn.LayerNorm,
                ):
        """ Initialize the IdeaDecoder model.
        Args:
            block_size (int): The maximum number of tokens for the decoder.
            vocab_size (int): The size of the vocabulary for the decoder.
            input_dim (int): The dimension of the input idea embeddings.
            decoder_embed_dim (int): The embedding dimension for the decoder.
            decoder_depth (int): The number of transformer blocks in the decoder.
            decoder_num_heads (int): The number of attention heads in each transformer block.
            mlp_ratio (float): The ratio of the hidden dimension to the embedding dimension in the MLP.
            norm_layer (nn.Module): The normalization layer to use.
        Returns:
            None
        """
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
        Forward pass for the IdeaDecoder model.
        Args:
            x (torch.Tensor): Input token indices of shape (B*I, T), where B is the batch size and I is the number of ideas.
            idea_vec (torch.Tensor): Input idea embeddings of shape (B*I, input_dim).
            target (torch.Tensor, optional): Target values for training. Not used in this implementation.
        Returns:
            logits (torch.Tensor): Logits for next-word prediction of shape (B*I, T, vocab_size).   
        """
        B, T = x.size()   # now B and T are ints

        # Project the idea embeddings
        idea_vec = self.idea_proj(idea_vec)

        print(f"Input x shape: {x.shape}")
        print(f"Idea vector shape: {idea_vec.shape}")   

        # forward the token and positional embeddings
        token_emb = self.decoder_wte(x)  # (B*I, T, decoder_embed_dim)
        pos_ids = torch.arange(0, T, device=x.device)
        pos_emb = self.decoder_wpe(pos_ids)       
        idea_token = idea_vec.unsqueeze(1)                 # (B*I, 1, D)
        token_and_pos = token_emb + pos_emb               # (B*I, T, D)
        hidden = torch.cat([idea_token, token_and_pos], 1) # (B*I, T+1, D)

        # idea token shape
        print(f"Idea token shape: {idea_token.shape}")
        print(f"Token and position embedding shape: {token_and_pos.shape}")
        print(f"Hidden shape after concatenation: {hidden.shape}")
        
        # Forward through decoder blocks
        for block in self.block:
            hidden = block(hidden)

        # Normalize the hidden states
        hidden = self.decoder_ln(hidden)

        # skip the idea token when producing logits for next-word prediction
        logits = self.decoder_lm_head(hidden[:, 1:, :])  # shape (B*I, T, vocab_size)

        return logits