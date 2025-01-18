import torch, tiktoken, csv, nltk
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import IterableDataset
from dataclasses import dataclass

from .text_preprocess import preprocess_text

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
        return 3400000

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
                yield preprocess_text(text, self.extended_enc, self.eos, self.T)

    def __iter__(self):
        """
        Yields text from each CSV shard in sequence.
        """
        for file_path in self.file_paths:
            yield from self.parse_csv_rows(file_path)


class ValCsvDataset(Dataset):
    """
    Loads CSV file for validation.
    """

    def __init__(self, file_path, max_length=77, text_column=0, skip_header=True):
        super().__init__()
        self.samples = []

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

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            if skip_header:
                next(reader, None)  # Skip header row if needed

            for i, row in enumerate(reader):
                if len(row) > text_column:
                    text = row[text_column]
                    self.samples.append(text)

    def __len__(self):
        return 7000

    def __getitem__(self, idx):
        text = self.samples[idx]
        return preprocess_text(text, self.extended_enc, self.eos, self.T)

# this is old data loader which is not streaming
# class SentenceDataset(torch.utils.data.Dataset):
#     def __init__(self, max_length=77):
#         super().__init__()
#         with open('data/input_lor.txt', 'r') as f:
#             text = f.read()
#         nltk.download('punkt')
#         self.sentences_raw = nltk.sent_tokenize(text)
#         self.eos = 50257
#         self.T = max_length

#         # Load the base encoding
#         enc = tiktoken.get_encoding("gpt2")
#         # Define new special tokens
#         new_special_tokens = {
#             "<|endofsent|>": self.eos,  # Make sure this ID does not conflict with existing tokens
#         }
#         # Create a new encoding with the added special tokens
#         self.extended_enc = tiktoken.Encoding(
#             name="gpt2_extended",
#             pat_str=enc._pat_str,  # Use the same pattern as the original encoding
#             mergeable_ranks=enc._mergeable_ranks,  # Keep the same mergeable ranks
#             special_tokens={**enc._special_tokens, **new_special_tokens},  # Extend special tokens
#         )

#     def __getitem__(self, idx):
#         sent = self.sentences_raw[idx]
#         return preprocess_text(sent, self.extended_enc, self.eos, self.T)
    
#     def __len__(self):
#         return len(self.sentences_raw)