import torch

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