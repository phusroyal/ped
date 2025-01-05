import os
import multiprocessing as mp
import numpy as np
import tiktoken
import nltk
import pandas as pd
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
local_dir = "wikitext"
remote_name = "wikitext-103-v1"
shard_size = int(1e6) # 100M sents per shard, total of 100 shards

nltk.download('punkt')

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("Salesforce/wikitext", name=remote_name, split="train")

def split_sent(doc):
    text = doc['text']
    return nltk.sent_tokenize(text)

def write_datafile(filename, sents_np):
    df = pd.DataFrame(sents_np)
    df.to_csv(filename, sep=',', index=False, encoding='utf-8')

nprocs = max(1, os.cpu_count()//2)
num_val = 5000
with mp.Pool(nprocs) as pool:
    shard_index = 0
    sent_count = 0
    all_sents_np = [0]*shard_size
    progress_bar = None

    for sents in pool.imap(split_sent, fw, chunksize=16):
        # is there enough space in the current shard for the new sents?
        if sent_count + len(sents) < shard_size:
            # simply append sents to current shard
            all_sents_np[sent_count:sent_count+len(sents)] = sents
            sent_count += len(sents)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="sents", desc=f"Shard {shard_index}")
            progress_bar.update(len(sents))
        else:
            # write the current shard and start a new one
            # split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"wikitext_train_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - sent_count
            progress_bar.update(remainder)
            all_sents_np[sent_count:sent_count+remainder] = sents[:remainder]
            write_datafile(filename, all_sents_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_sents_np[0:len(sents)-remainder] = sents[remainder:]
            sent_count = len(sents)-remainder
    
    # write any remaining tokens as the last shard
    if sent_count != 0:
        # split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"wikitext_train_{shard_index:06d}")
        write_datafile(filename, all_sents_np[:sent_count-num_val])
        filename = os.path.join(DATA_CACHE_DIR, f"wikitext_val_{shard_index:06d}")
        write_datafile(filename, all_sents_np[sent_count-num_val:sent_count])