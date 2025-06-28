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
from sentence_transformers import SentenceTransformer

BASE_DIR = '/home/anwoy/phuhoang/ped/'

class PretrainedIdearEncoder(nn.Module):
    def __init__(self, 
                 block_size = 32, # batch size
                 idea_size = 8, # number of ideas per batch
                st_model_name='sentence-transformers/all-mpnet-base-v2',
                 device='cpu',
                 local_file_only=True,
                 torch_dtype=torch.float32,
                 ):
        super().__init__()
        """Initialize the PretrainedIdeaEncoder model.
        Args:
            st_model_name (str): The name of the Sentence Transformer model to use.
            device (str): The device to run the model on ('cpu' or 'cuda').
            local_file_only (bool): Whether to only use local files for the model.
            torch_dtype (torch.dtype): The data type for the model weights.
        """

        # Load the Sentence Transformer model
        self.device = device
        self.B = block_size
        self.I = idea_size  # Number of ideas per batch
        cache_folder = os.path.join(BASE_DIR, 'saved_models', st_model_name)

        self.sbert = SentenceTransformer(st_model_name,
                                         device=device,
                                        cache_folder=cache_folder,
                                        local_files_only=local_file_only,
                                        model_kwargs={
                                            'torch_dtype': torch_dtype
                                        })
        
        # Set the model to evaluation mode
        self.sbert.eval()

    def get_sentence_embedding_dimension(self):                                 
        # The output dimension is typically 768
        return self.sbert.get_sentence_embedding_dimension()

    def forward(self, sentence_list, convert_to_tensor=True):
        # Get the idea embeddings from all-mpnet-base-v2
        """Forward pass for the PretrainedIdeaEncoder model.
        Args:
            sentence_list (list of str): List of sentences to encode.
        Returns:
            torch.Tensor: Idea embeddings of shape (B, sbert_dim), where B is the batch size, and sbert_dim is the dimension of the sentence embeddings (typically 768).
        """

        with torch.no_grad():
            idea_vecs = self.sbert.encode(sentence_list, 
                                          convert_to_tensor=convert_to_tensor)  # shape (B, 768)
            
        # Reshape the idea vectors to (B, I, sbert_dim)
        idea_vecs = idea_vecs.view(self.B, self.I, -1)
        return idea_vecs.to(self.device)