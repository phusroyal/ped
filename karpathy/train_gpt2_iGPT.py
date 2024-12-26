import os, sys, math, time, inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import nltk
import tiktoken
from transformers import BertTokenizer, BertModel
from dataclasses import dataclass

#-----
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
# DDP launch for e.g. 2 GPUs:
# torchrun --standalone --nproc_per_node=2 train_gpt2_shake.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# -----------------------------------------------------------------------------
# load data
from utils.dataloaderlite import SentenceDataLoaderLite, SentenceDataset, preprocessText

# total_batch_size = 524288
# B = 16
# T = 1024
# assert total_batch_size % (B * T * ddp_world_size) == 0
# grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
# if master_process:
#     print(f"total desired batch size: {total_batch_size}")
#     print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# torch.set_float32_matmul_precision('high')
# enc = tiktoken.get_encoding('gpt2')
# bert_tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-small')
# bert_model = BertModel.from_pretrained('prajjwal1/bert-small')
# bert_model.eval()
# dataset = SentenceDataset("input.txt", enc, bert_tokenizer, bert_model)
# train_loader = SentenceDataLoaderLite(dataset, block_size=T, B=B, process_rank=ddp_rank, num_processes=ddp_world_size)

B = 1
T = 256
total_batch_size = B*T #524288
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
torch.set_float32_matmul_precision('high')


# Load the base encoding
enc = tiktoken.get_encoding("gpt2")
eos = 50257
# Define new special tokens
new_special_tokens = {
    "<|endofsent|>": eos,  # Make sure this ID does not conflict with existing tokens
}
# Create a new encoding with the added special tokens
extended_enc = tiktoken.Encoding(
    name="gpt2_extended",
    pat_str=enc._pat_str,  # Use the same pattern as the original encoding
    mergeable_ranks=enc._mergeable_ranks,  # Keep the same mergeable ranks
    special_tokens={**enc._special_tokens, **new_special_tokens},  # Extend special tokens
)

# dataset = SentenceDataset("data/input.txt", enc)
# train_loader = SentenceDataLoaderLite(eos= eos, enc=extended_enc, block_size=T, B=B)

def infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

train_dataset = SentenceDataset()
train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True)
train_loader = infinite_loader(train_loader)
# -----------------------------------------------------------------------------
# create model
from model.unifiedGPT import UnifiedGPT, UnifiedGPTConfig

# main_gpt_config = GPTConfig(block_size=T, vocab_size=50304, n_layer=8, n_head=12, n_embd=768)
# idea_gpt_config = iGPTConfig(block_size=T, vocab_size=50304, n_layer=4, n_head=8, n_embd=512, idea_dim=768)

# main_gpt_config = GPTConfig(block_size=T, vocab_size=50304, n_layer=4, n_head=8, n_embd=256)
# idea_gpt_config = iGPTConfig(block_size=T, vocab_size=50304, n_layer=2, n_head=4, n_embd=128, idea_dim=256)

# main_gpt = mainGPT(main_gpt_config)
# idea_gpt = iGPT(idea_gpt_config)
# model = DualGPT(main_gpt, idea_gpt).to(device)
# model = torch.compile(model)

# Example usage:
config = UnifiedGPTConfig(
    block_size=256,
    vocab_size=50304,
    n_layer_main=4,
    n_layer_idea=2,
    n_head=8,
    n_embd_main = 768,          
    n_embd_idea = 512,
    idea_dim=768,
)
model = UnifiedGPT(config).to(device)
# model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
total_params = sum(p.numel() for p in raw_model.parameters())
print(f"Number of parameters: {total_params}")

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 500
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        # x, y, ix, sent = train_loader.next_batch()
        sent = next(train_loader)
        print(sent)
        x, y, ix = preprocessText(sent, extended_enc, eos, T)
        x, y, ix = x.to(device), y.to(device), ix.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, ix, y)
        
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = B * T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")


if ddp:
    destroy_process_group()

torch.save(model.state_dict(), "model_checkpoint.pth")


import sys; sys.exit(0)

# prefix tokens
model.eval()
num_return_sequences = 5
max_length = 30
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)