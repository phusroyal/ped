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

from model.idea_encoder import PretrainedIdearEncoder
from model.idea_predictor import IdeaPredictor
from model.idea_decoder import IdeaDecoder

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
        self.idea_decoder_block_size = config.id_dec_block_size

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

        # flatten to (B*I, predictor_embed_dim)
        ys = ys.view(-1, self.sbert_dim)
        print(f"ys shape after flattening: {ys.shape}")  # Debugging line

        # Pass to idea decoder
        logits = self.idea_decoder(xt, ys)

        return logits, ys, idea_vecs



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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# set precision high
torch.set_float32_matmul_precision('high')  # Set precision for matmul operations

# Initialize model
model = iGPT(test_config,
             device=device, 
             torch_dtype=torch.bfloat16,)

# Create sample inputs
sample_tokens = torch.randint(0, 50304, (10, 16)).to(device)  # 10 sequences of length 16
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
    logits, ys, idea_vecs = model(sample_tokens, sample_sentences)

print(f"Output logits shape: {logits.shape}")
print(f"Output ys shape: {ys.shape}")
print(f"Idea vectors shape: {idea_vecs.shape}")
    
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
