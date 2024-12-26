import torch
import torch.utils
from torch.utils.data import Dataset
import torch.utils.data
import nltk
from dataclasses import dataclass

# -----------------------------------------------------------------------------

# class SentenceDataset(Dataset):
#     def __init__(self, text_path, enc):
#         with open(text_path, 'r', encoding='utf-8') as f:
#             text = f.read()
#         nltk.download('punkt')
#         self.sentences_raw = nltk.sent_tokenize(text)
        
#         self.sentences = [s + " <|endofsent|>" for s in self.sentences_raw]
#         self.processed_text = " ".join(self.sentences)
#         self.enc = enc
#         self.tokens = torch.tensor(enc.encode(self.processed_text), dtype=torch.long)

#     def __len__(self):
#         return len(self.tokens)

#     def __getitem__(self, idx):
#         return self.tokens[idx]

class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        with open('data/input_lor.txt', 'r') as f:
            text = f.read()
        nltk.download('punkt')
        self.sentences_raw = nltk.sent_tokenize(text)
    
    def __getitem__(self, idx):
        return self.sentences_raw[idx]
    
    def __len__(self):
        return len(self.sentences_raw)


def preprocessText(sent, enc, eos, max_length=77):
    B = len(sent)
    T = max_length

    x = torch.zeros((B, T), dtype=torch.long)
    y = torch.zeros((B, T), dtype=torch.long)
    ix = torch.zeros((B, T), dtype=torch.long)

    for idx in range(len(sent)):
        s = sent[idx]
        encoded_sent = enc.encode(s)
        encoded_sent.insert(0, eos)  # Add <eos> token at the start
        encoded_sent.append(eos)    # Add <eos> token at the end
        tokens = torch.tensor(encoded_sent, dtype=torch.long)
            
        # Ensure the tokens fit in the fixed sequence length T
        tokens = tokens[:T+1]  # Truncate if tokens exceed T+1 (account for <eos>)

        # Assign to x and y with proper slicing
        x[idx, :len(tokens)-1] = tokens[:-1]
        y[idx, :len(tokens)-1] = tokens[1:]
        ix[idx, :len(tokens)-2] = tokens[1:-1]
    
    return x, y, ix


class SentenceDataLoaderLite:
    def __init__(self, eos, enc, block_size, B, process_rank=0, num_processes=1):

        with open('data/input_lor.txt', 'r') as f:
            text = f.read()
        nltk.download('punkt')
        self.sentences_raw = nltk.sent_tokenize(text)
        print(self.sentences_raw)
        self.eos = eos
        self.enc = enc

        self.B = B
        self.T = block_size

        self.process_rank = process_rank
        self.num_processes = num_processes
        # self.current_position = B * block_size * process_rank
        self.length = len(self.sentences_raw)

    def next_batch(self):
        B, T = self.B, self.T
        buf_sent = self.sentences_raw[self.current_position : self.current_position + B]
        
        assert len(buf_sent) == B, f"Expected {B} sentences but got {len(buf_sent)}"

        x = torch.zeros((B, T), dtype=torch.long)
        y = torch.zeros((B, T), dtype=torch.long)
        ix = torch.zeros((B, T), dtype=torch.long)
        
        for idx, sent in enumerate(buf_sent):
            print(sent)
            encoded_sent = self.enc.encode(sent)
            encoded_sent.insert(0, self.eos)  # Add <eos> token at the start
            encoded_sent.append(self.eos)    # Add <eos> token at the end
            tokens = torch.tensor(encoded_sent, dtype=torch.long)
            
            # Ensure the tokens fit in the fixed sequence length T
            tokens = tokens[:T+1]  # Truncate if tokens exceed T+1 (account for <eos>)
            
            # Assign to x and y with proper slicing
            x[idx, :len(tokens)-1] = tokens[:-1]
            y[idx, :len(tokens)-1] = tokens[1:]
            ix[idx, :len(tokens)-2] = tokens[1:-1]
        
        self.current_position += B * self.num_processes
        if self.current_position + B > self.length:  # or some criterion
            self.current_position = self.B * self.process_rank
        # self.current_position += B * T * self.num_processes
        # if self.current_position + (B * T * self.num_processes + 1) > self.length:
        #     self.current_position = self.B * self.T * self.process_rank

        return x, y, ix, buf_sent
