import torch
import torch.utils
from torch.utils.data import Dataset
import torch.utils.data
import nltk
from dataclasses import dataclass
import tiktoken


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

def preprocessText(sent, enc, eos, max_length=77):
    T = max_length

    x = torch.zeros((T), dtype=torch.long)
    y = torch.zeros((T), dtype=torch.long)
    ix = torch.zeros((T), dtype=torch.long)

    encoded_sent = enc.encode(sent)
    encoded_sent.insert(0, eos)  # Add <eos> token at the start
    encoded_sent.append(eos)    # Add <eos> token at the end
    tokens = torch.tensor(encoded_sent, dtype=torch.long)
        
    # Ensure the tokens fit in the fixed sequence length T
    tokens = tokens[:T+1]  # Truncate if tokens exceed T+1 (account for <eos>)

    # Assign to x and y with proper slicing
    x[:len(tokens)-1] = tokens[:-1]
    y[:len(tokens)-1] = tokens[1:]
    ix[:len(tokens)-2] = tokens[1:-1]
    
    return x, y, ix


class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, max_length=77):
        super().__init__()
        with open('data/input_lor.txt', 'r') as f:
            text = f.read()
        nltk.download('punkt')
        self.sentences_raw = nltk.sent_tokenize(text)
        self.eos = 50257
        self.T = max_length

        # Load the base encoding
        enc = tiktoken.get_encoding("gpt2")
        # Define new special tokens
        new_special_tokens = {
            "<|endofsent|>": self.eos,  # Make sure this ID does not conflict with existing tokens
        }
        # Create a new encoding with the added special tokens
        self.extended_enc = tiktoken.Encoding(
            name="gpt2_extended",
            pat_str=enc._pat_str,  # Use the same pattern as the original encoding
            mergeable_ranks=enc._mergeable_ranks,  # Keep the same mergeable ranks
            special_tokens={**enc._special_tokens, **new_special_tokens},  # Extend special tokens
        )

    def __getitem__(self, idx):
        sent = self.sentences_raw[idx]
        return preprocessText(sent, self.extended_enc, self.eos, self.T)
    
    def __len__(self):
        return len(self.sentences_raw)


import csv
import torch
from torch.utils.data import IterableDataset

class StreamingCsvDataset(IterableDataset):
    """
    An IterableDataset that streams CSV rows from multiple files,
    line-by-line, to avoid loading everything into memory.

    Args:
        file_paths (List[str]): Paths to CSV files.
        text_column (int): Which column index contains the text (0-based).
        skip_header (bool): If True, skip the first row (header) in each CSV.
        transform (callable, optional): Optional transform (e.g., tokenization)
            applied to each line. Defaults to None.

    Example CSV format:
        col1, col2, col3, ...
        ...
    """
    def __init__(self, file_paths, max_length=77, text_column=0, skip_header=True):
        super().__init__()
        self.file_paths = file_paths
        self.text_column = text_column
        self.skip_header = skip_header

        self.eos = 50257
        self.T = max_length

        # Load the base encoding
        enc = tiktoken.get_encoding("gpt2")
        # Define new special tokens
        new_special_tokens = {
            "<|endofsent|>": self.eos,  # Make sure this ID does not conflict with existing tokens
        }
        # Create a new encoding with the added special tokens
        self.extended_enc = tiktoken.Encoding(
            name="gpt2_extended",
            pat_str=enc._pat_str,  # Use the same pattern as the original encoding
            mergeable_ranks=enc._mergeable_ranks,  # Keep the same mergeable ranks
            special_tokens={**enc._special_tokens, **new_special_tokens},  # Extend special tokens
        )

    def __len__(self):
        return 3700000

    def parse_csv_rows(self, file_path):
        """
        Generator that yields lines from one CSV file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            if self.skip_header:
                next(reader, None)  # skip header row
            for row in reader:
                if len(row) <= self.text_column:
                    # Skip rows that don't have enough columns
                    continue
                text = row[self.text_column]
                yield preprocessText(text, self.extended_enc, self.eos, self.T)

    def __iter__(self):
        """
        Yields text from each CSV shard in sequence.
        """
        for file_path in self.file_paths:
            yield from self.parse_csv_rows(file_path)

# class SentenceDataLoaderLite:
#     def __init__(self, eos, enc, block_size, B, process_rank=0, num_processes=1):

#         with open('data/input_lor.txt', 'r') as f:
#             text = f.read()
#         nltk.download('punkt')
#         self.sentences_raw = nltk.sent_tokenize(text)
#         # print(self.sentences_raw)
#         self.eos = eos
#         self.enc = enc

#         self.B = B
#         self.T = block_size

#         self.process_rank = process_rank
#         self.num_processes = num_processes
#         # self.current_position = B * block_size * process_rank
#         self.length = len(self.sentences_raw)

#     def next_batch(self):
#         B, T = self.B, self.T
#         buf_sent = self.sentences_raw[self.current_position : self.current_position + B]
        
#         assert len(buf_sent) == B, f"Expected {B} sentences but got {len(buf_sent)}"

#         x = torch.zeros((B, T), dtype=torch.long)
#         y = torch.zeros((B, T), dtype=torch.long)
#         ix = torch.zeros((B, T), dtype=torch.long)
        
#         for idx, sent in enumerate(buf_sent):
#             # print(sent)
#             encoded_sent = self.enc.encode(sent)
#             encoded_sent.insert(0, self.eos)  # Add <eos> token at the start
#             encoded_sent.append(self.eos)    # Add <eos> token at the end
#             tokens = torch.tensor(encoded_sent, dtype=torch.long)
            
#             # Ensure the tokens fit in the fixed sequence length T
#             tokens = tokens[:T+1]  # Truncate if tokens exceed T+1 (account for <eos>)
            
#             # Assign to x and y with proper slicing
#             x[idx, :len(tokens)-1] = tokens[:-1]
#             y[idx, :len(tokens)-1] = tokens[1:]
#             ix[idx, :len(tokens)-2] = tokens[1:-1]
        
#         self.current_position += B * self.num_processes
#         if self.current_position + B > self.length:  # or some criterion
#             self.current_position = self.B * self.process_rank
#         # self.current_position += B * T * self.num_processes
#         # if self.current_position + (B * T * self.num_processes + 1) > self.length:
#         #     self.current_position = self.B * self.T * self.process_rank

#         return x, y, ix, buf_sent
