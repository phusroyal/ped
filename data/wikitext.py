import os
import multiprocessing as mp
import numpy as np
import spacy
import pandas as pd
from datasets import load_dataset  
from tqdm import tqdm            

#[train] Total sentences processed: 3498072
#[val] Total sentences processed: 6987
# ------------------------------------------
local_dir = "wikitext"
remote_name = "wikitext-103-v1"
shard_size = int(1e6)  # e.g. 1 million sents per shard
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

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

def is_valid_sentence(s: str) -> bool:
    """
    Returns True if sentence s satisfies all the filters:
      1. does not contain '= =' or '<unk>'
      2. has length >= 10 characters
    """
    # skip if s contains "= =" or "<unk>"
    if "= =" in s or "<unk>" in s:
        return False
    if s.startswith('=') and s.endswith('='):
        return False
    # skip if length < 10
    if len(s) < 10:
        return False
    return True

def split_sent(doc):
    """
    spaCy-based sentence splitter + filter.
    doc: a dictionary like {'text': ...}
    returns a list of sentence strings
    """
    init_spacy()  # ensure spaCy model is loaded in this process
    text = doc["text"]
    if not text.strip():
        return []
    spacy_doc = _nlp(text)

    # Collect each sentence's text, stripping whitespace
    sents = [sent.text.strip() for sent in spacy_doc.sents if sent.text.strip()]

    # Apply custom filters
    sents = [s for s in sents if is_valid_sentence(s)]
    return sents

def write_datafile(filename, sents_list):
    """
    Writes a list of sentences to a CSV file using pandas.
    """
    df = pd.DataFrame(sents_list)
    df.to_csv(filename, sep=",", index=False, encoding="utf-8")

def flush_shard_to_csv(shard_sents, shard_idx, prefix):
    """
    Writes out the current shard to a CSV file. Example filename:
      wikitext_train_000000
      or
      wikitext_val_000000
    """
    filename = os.path.join(DATA_CACHE_DIR, f"wikitext_{prefix}_{shard_idx:06d}")
    write_datafile(filename, shard_sents)
    return filename

def process_dataset(dataset, prefix="train"):
    """
    Processes an entire dataset (train or val) in shards of size 'shard_size'.
    All data in 'dataset' is written to wikitext_{prefix}_XXXXX files.

    No manual slicing of last N sentences.
    """
    nprocs = max(1, os.cpu_count() // 2)
    total_sent = 0
    shard_index = 0
    sent_count = 0
    all_sents_np = [None] * shard_size  # buffer for the current shard
    progress_bar = None

    # Create a multiprocessing pool
    with mp.Pool(nprocs) as pool:
        # Iterate over dataset in parallel, splitting text into sentences
        for sents in pool.imap(split_sent, dataset, chunksize=16):
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
                    progress_bar = tqdm(total=shard_size, unit="sents",
                                        desc=f"{prefix} Shard {shard_index}")
                progress_bar.update(len(sents))
            else:
                # We need to fill the current shard and then flush it
                remainder = shard_size - sent_count
                if progress_bar is not None:
                    progress_bar.update(remainder)
                    progress_bar.close()
                    progress_bar = None

                # Fill up the current shard
                all_sents_np[sent_count : sent_count + remainder] = sents[:remainder]
                # Write out the shard
                flush_shard_to_csv(all_sents_np, shard_index, prefix=prefix)
                shard_index += 1

                # leftover goes into next shard
                leftover_count = len(sents) - remainder
                all_sents_np[: leftover_count] = sents[remainder:]
                sent_count = leftover_count

                # Create a new progress bar for the next shard if leftover is non-empty
                if leftover_count > 0:
                    progress_bar = tqdm(total=shard_size, unit="sents",
                                        desc=f"{prefix} Shard {shard_index}")
                    progress_bar.update(leftover_count)

        # Done reading all data, flush last partial shard if it has content
        if sent_count > 0:
            if progress_bar is not None:
                progress_bar.close()
                progress_bar = None

            # Write out the partial shard
            flush_shard_to_csv(all_sents_np[:sent_count], shard_index, prefix=prefix)

    print(f"[{prefix}] Total sentences processed:", total_sent)

def main():
    # 1) Load the train split
    train_fw = load_dataset("Salesforce/wikitext", name=remote_name, split="train")
    # 2) Load the val split
    val_fw = load_dataset("Salesforce/wikitext", name=remote_name, split="validation")

    # Process each split separately
    process_dataset(train_fw, prefix="train")
    process_dataset(val_fw, prefix="val")

if __name__ == "__main__":
    main()