"""
This is iGPT v0.1.0
- Encoder is sentence-transformers/all-mpnet-base-v2 (dim 768) (https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
- Decoder is GPT (from scratch)
- ivec is treated as a prepended token
"""

# %%

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Or more simply:
sys.path.append('..')

os.environ["https_proxy"] = "http://xen03.iitd.ac.in:3128"
os.environ["http_proxy"] = "http://xen03.iitd.ac.in:3128"

import torch
import torch.nn as nn
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

from model.transformer_blocks import Block
from model.idea_encoder import PretrainedIdearEncoder

BASE_DIR = '/home/anwoy/phuhoang/ped/'

@dataclass
class iGPTConfig:
    id_enc_block_size: int = 2  
    id_enc_idea_size: int = 5  

    id_pred_block_size: int = 32
    id_pred_n_layer: int = 6
    id_pred_n_head: int = 12
    id_pred_n_embd: int = 384

    id_dec_block_size: int = 32
    id_dec_n_layer: int = 6
    id_dec_n_head: int = 12
    id_dec_n_embd: int = 384
    
    mlp_ratio: float = 4
    vocab_size: int = 50304


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
        pos_ids = torch.arange(0, B, device=idea_vecs.device)  # length T
        print(f"pos_ids shape: {pos_ids.shape}")  # Debugging line
        print(f"pos_ids: {pos_ids}")  # Debugging line
        a
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

        # Normalize the hidden states
        hidden = self.decoder_ln(hidden)

        # skip the idea token when producing logits for next-word prediction
        logits = self.decoder_lm_head(hidden[:, 1:, :])  # shape (B*I, T, vocab_size)

        return logits


class iGPT(nn.Module):
    """
    GPT model that prepends a single 'idea token' derived from
    all-mpnet-base-v2 embeddings at the start of the sequence.
    """

    def __init__(self, config: iGPTConfig,
                 device='cpu',
                 torch_dtype=torch.float32,
                 st_model_name='sentence-transformers/all-mpnet-base-v2'):
        super().__init__()
        self.config = config
        self.device = device
        self.torch_dtype = torch_dtype

        # -- idea encoder
        self.sbert = PretrainedIdearEncoder(
            block_size=config.id_enc_block_size,
            idea_size=config.id_enc_idea_size,

            st_model_name=st_model_name,
            device=device,
            local_file_only=True,
            torch_dtype=torch_dtype
        )
        self.sbert_dim = self.sbert.get_sentence_embedding_dimension()  # typically 768
        # print device of sbert
        print(f"Sentence Transformer model loaded on device: {self.sbert.device}")
        
        # -- idea predictor
        self.idea_predictor = IdeaPredictor(
            block_size=config.id_pred_block_size,
            input_dim=self.sbert_dim,
            predictor_embed_dim=config.id_pred_n_embd,
            predictor_depth=config.id_pred_n_layer,
            predictor_num_heads=config.id_pred_n_head,
            mlp_ratio=config.mlp_ratio,
            norm_layer=nn.LayerNorm
        )
        self.idea_predictor.to(device=device, dtype=torch_dtype)

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
        self.idea_decoder.to(device=device, dtype=torch_dtype)

    def forward(self, xt, xs):
        """
        Forward pass for the iGPT model.
        Args:
            xt (torch.Tensor): Input token indices of shape (B*I, T), where B is the batch size and I is the number of ideas.
            xs (list of str): List of sentences corresponding to each idea in the batch.
        Returns:
            yt (torch.Tensor): Logits for next-word prediction of shape (B*I, T, vocab_size).
            ys (torch.Tensor): Idea embeddings of shape (B*I, predictor_embed_dim).
            idea_vecs (torch.Tensor): Idea embeddings from the Sentence Transformer of shape (B, I, 768).
        """
        # B, I, T = xt.size(), xt.size(1), xt.size(2)

        print(f"xt shape: {xt.shape}, xs length: {len(xs)}")  # Debugging line
        print(f"xt: {xt}, xs: {xs}")  # Debugging line

        # Get the idea embeddings from all-mpnet-base-v2 for all sentences I in the batch
        with torch.no_grad():
            idea_vecs = self.sbert(xs, convert_to_tensor=True)  # shape (B, I, 768)
        
        print(f"idea_vecs shape: {idea_vecs.shape}")  # Debugging line
        # print device of idea_vecs
        print(f"Idea embeddings loaded on device: {idea_vecs.device}") # Debugging line

        # Pass to idea predictor
        ys = self.idea_predictor(idea_vecs) # shape (B, I, predictor_embed_dim)
        print(f"ys shape after predictor: {ys.shape}")  # Debugging line
        a

        # flatten to (B*I, predictor_embed_dim)
        ys = ys.view(B*I, -1)

        # Pass to idea decoder
        yt = self.idea_decoder(xt, ys)

        return yt, ys, idea_vecs



# %%
# def test_igpt():
    # Small test configuration

test_config = iGPTConfig(
    id_enc_block_size=2,
    id_enc_idea_size=5,
    
    id_pred_block_size=32,
    id_pred_n_layer=2,
    id_pred_n_head=4,
    id_pred_n_embd=128,

    id_dec_block_size=32,
    id_dec_n_layer=2,
    id_dec_n_head=4,
    id_dec_n_embd=128,

    mlp_ratio=4,
    vocab_size=50304
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# set precision high
torch.set_float32_matmul_precision('high')  # Set precision for matmul operations

# Initialize model
model = iGPT(test_config,
             device=device, 
             torch_dtype=torch.bfloat16,)

# Create sample inputs
sample_tokens = torch.randint(0, 50304, (2, 16))  # 2 sequences of length 16
# sample 10  sentences for the idea encoder
sample_sentences =  [
    "This is the first sentence.",
    "This is the second sentence.",
    "This is the third sentence.",
    "This is the fourth sentence.",
    "This is the fifth sentence.", 
    "This is the sixth sentence.",
    "This is the seventh sentence.",
    "This is the eighth sentence.",
    "This is the ninth sentence.",
    "This is the tenth sentence."
]

# Forward pass
with torch.no_grad():
    logits, embeddings, _ = model(sample_tokens, sample_sentences)

print(f"Output logits shape: {logits.shape}")
print(f"Output embeddings shape: {embeddings.shape}")
    
    # return model


# # %%
# # ping google
# import requests
# def ping_google():
#     try:
#         response = requests.get('https://huggingface.co', timeout=5)
#         if response.status_code == 200:
#             print("Google is reachable.")
#         else:
#             print(f"Google returned status code: {response.status_code}")
#     except requests.RequestException as e:
#         print(f"Error reaching Google: {e}")
# ping_google()

# # %%
