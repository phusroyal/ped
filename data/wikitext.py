import os
import multiprocessing as mp
import numpy as np
import spacy
import pandas as pd
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm             # pip install tqdm

# ------------------------------------------
local_dir = "wikitext"
remote_name = "wikitext-103-v1"
shard_size = int(1e6)  # e.g. 1 million sents per shard
num_val = 5000         # how many sentences to reserve for validation in the last shard
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Download the dataset
fw = load_dataset("Salesforce/wikitext", name=remote_name, split="train")

# We'll create a global variable for spaCy model, but load it lazily in each process
_nlp = None

def init_spacy():
    """
    Helper to initialize the spaCy model in each worker.
    """
    global _nlp
    if _nlp is None:
        # load a small pipeline, disable components not needed for sentence segmentation
        _nlp = spacy.load("en_core_web_sm", disable=["tagger", "ner", "lemmatizer"])
        # ensure the sentencizer/pipeline can segment sentences
        if not _nlp.has_pipe("sentencizer"):
            _nlp.add_pipe("sentencizer")

def split_sent(doc):
    """
    spaCy-based sentence splitter.
    doc: a dictionary like {'text': ...}
    returns a list of sentence strings
    """
    init_spacy()  # ensure spaCy model is loaded in this process
    text = doc["text"]
    if not text.strip():
        # If there's no text, return an empty list
        return []
    spacy_doc = _nlp(text)
    # collect each sentence's text, stripping whitespace
    sents = [sent.text.strip() for sent in spacy_doc.sents if sent.text.strip()]
    return sents

def write_datafile(filename, sents_list):
    """
    Writes a list of sentences to a CSV file using pandas.
    """
    df = pd.DataFrame(sents_list)
    df.to_csv(filename, sep=",", index=False, encoding="utf-8")

def flush_shard_to_csv(shard_sents, shard_idx, is_train=True):
    """
    Writes out the current shard to a CSV file, then returns the filename written.
    """
    split_name = "train" if is_train else "val"
    filename = os.path.join(DATA_CACHE_DIR, f"wikitext_{split_name}_{shard_idx:06d}")
    write_datafile(filename, shard_sents)
    return filename

def process_shard(all_sents_np, sents_in_shard, shard_idx):
    """
    Writes the full shard as 'train' CSV, increments shard_idx.
    Returns updated shard_idx.
    """
    flush_shard_to_csv(all_sents_np, shard_idx, is_train=True)
    return shard_idx + 1

def main():
    nprocs = max(1, os.cpu_count() // 2)
    total_sent = 0
    shard_index = 0
    sent_count = 0
    all_sents_np = [None] * shard_size  # buffer for the current shard
    progress_bar = None

    with mp.Pool(nprocs) as pool:
        # Iterate over dataset in parallel, splitting text into sentences
        for sents in pool.imap(split_sent, fw, chunksize=16):
            if not sents:
                # skip empty results
                continue
            total_sent += len(sents)

            # Check if there's enough space in the current shard
            if sent_count + len(sents) <= shard_size:
                # Append sents to current shard
                all_sents_np[sent_count : sent_count + len(sents)] = sents
                sent_count += len(sents)
                # Update or create progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="sents", desc=f"Shard {shard_index}")
                progress_bar.update(len(sents))
            else:
                # We need to fill the current shard and then flush it
                remainder = shard_size - sent_count
                if remainder < 0:
                    # This would be a sign of something going wrong (shouldn't happen).
                    raise ValueError(f"Invalid remainder: {remainder}. 'shard_size' or logic error?")

                # Fill up the current shard
                if progress_bar is not None:
                    progress_bar.update(remainder)
                    progress_bar.close()  # gracefully close the bar for this shard
                    progress_bar = None

                all_sents_np[sent_count : sent_count + remainder] = sents[:remainder]
                # Write out the shard
                shard_index = process_shard(all_sents_np, shard_size, shard_index)

                # leftover goes into next shard
                leftover_count = len(sents) - remainder
                all_sents_np[: leftover_count] = sents[remainder:]
                sent_count = leftover_count

                # Create a new progress bar for the next shard if leftover is non-empty
                if leftover_count > 0:
                    progress_bar = tqdm(total=shard_size, unit="sents", desc=f"Shard {shard_index}")
                    progress_bar.update(leftover_count)

        # Done reading all data, flush last partial shard
        if sent_count > 0:
            # Close any open progress bar
            if progress_bar is not None:
                progress_bar.close()
                progress_bar = None

            # For demonstration, we split the last shard into train/val
            # The last 'num_val' sentences become val, the rest train
            if sent_count <= num_val:
                # If we have fewer sentences than num_val, we store them all as val
                flush_shard_to_csv(all_sents_np[:sent_count], shard_index, is_train=False)
            else:
                flush_shard_to_csv(all_sents_np[: sent_count - num_val], shard_index, is_train=True)
                flush_shard_to_csv(all_sents_np[sent_count - num_val : sent_count], shard_index, is_train=False)

    print("Total sentences processed:", total_sent)

if __name__ == "__main__":
    main()
