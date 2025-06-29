import torch.nn.functional as F
from torch import nn
from model.transformer_blocks import Block
import torch

class IdeaPredictor(nn.Module):
    def __init__(self, 
                block_size=32,
                input_dim=768,
                predictor_embed_dim=384,
                predictor_depth=12,
                predictor_num_heads=12,
                mlp_ratio=4,
                # qkv_bias=False,
                # qk_scale=None,
                # drop_rate=0.0,
                # attn_drop_rate=0.0,
                # drop_path_rate=0.0,
                # init_std=0.02,
                norm_layer=nn.LayerNorm,
                ):
        """
        Initialize the IdeaPredictor model.
        Args:
            block_size (int): The maximum number of sequences for the predictor.
            input_dim (int): The dimension of the input idea embeddings.
            predictor_embed_dim (int): The embedding dimension for the predictor.
            predictor_depth (int): The number of transformer blocks in the predictor.
            predictor_num_heads (int): The number of attention heads in each transformer block.
            mlp_ratio (float): The ratio of the hidden dimension to the embedding dimension in the MLP.
            norm_layer (nn.Module): The normalization layer to use.
        Returns:
            None
        """
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
        # what and why is weight tying?
        # what: Weight tying is a technique where the same weight matrix is used for both the input and output projections in a model, reducing the number of parameters and potentially improving generalization.
        # why: This can help prevent overfitting, especially in models with large vocabularies or embedding spaces, by forcing the model to learn a more compact representation.
        # as idea prediction is a regression task, we do not use weight tying

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
        Forward pass for the IdeaPredictor model.
        Args:
            idea_vecs (torch.Tensor): Input idea embeddings of shape (B, I, input_dim), where B is the batch size and I is the number of ideas.
            target (torch.Tensor, optional): Target values for training. Not used in this implementation.
        """
        B, I = idea_vecs.size(0), idea_vecs.size(1)

        # Project the idea embeddings
        idea_emb = self.idea_proj(idea_vecs)  # shape (B, I, predictor_embed_dim)

        # Add positional embeddings
        pos_ids = torch.arange(0, I, dtype=torch.long, device=idea_vecs.device)  # length T
        pos_emb = self.predictor_wpe(pos_ids)  # shape  (B, I, predictor_embed_dim)
        hidden = idea_emb + pos_emb  # (B, I, predictor_embed_dim)

        # Forward through predictor blocks
        for block in self.predictor_blocks:
            hidden = block(hidden)
        hidden = self.predictor_ln(hidden)

        # Project back to the original dimension
        hidden = self.predictor_proj(hidden)

        return hidden