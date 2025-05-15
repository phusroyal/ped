import torch

def preprocess_text(sent, enc, max_length=32):
    """
        Text preprocess for iGPT v011
        Return:
            x: torch.Tensor (T)
            y: torch.Tensor (T)
            sent: str
    """
    T = max_length

    x = torch.zeros((T), dtype=torch.long)
    y = torch.zeros((T), dtype=torch.long)

    encoded_sent = enc.encode(sent)
    encoded_sent.append(enc.eot_token)    # Add <eot> token at the end
    tokens = torch.tensor(encoded_sent, dtype=torch.long)
        
    # Ensure the tokens fit in the fixed sequence length T
    tokens = tokens[:T+1]  # Truncate if tokens exceed T+1 (account for <eot>)

    # Assign to x and y with proper slicing
    x[:len(tokens)-1] = tokens[:-1] # x = [a,b,c], eos as sent emb will be added at position 0. Later, x = [sent_emb,a,b,c]
    y[:len(tokens)-1] = tokens # y = [a,b,c,eos]
    
    return x, y, sent

# def preprocessText(sent, enc, eos, max_length=77):
#     """
#         Text preprocess for iGPT v000
#     """
#     T = max_length

#     x = torch.zeros((T), dtype=torch.long)
#     y = torch.zeros((T), dtype=torch.long)
#     ix = torch.zeros((T), dtype=torch.long)

#     encoded_sent = enc.encode(sent)
#     encoded_sent.insert(0, eos)  # Add <eos> token at the start
#     encoded_sent.append(eos)    # Add <eos> token at the end
#     tokens = torch.tensor(encoded_sent, dtype=torch.long)
        
#     # Ensure the tokens fit in the fixed sequence length T
#     tokens = tokens[:T+1]  # Truncate if tokens exceed T+1 (account for <eos>)

#     # Assign to x and y with proper slicing
#     x[:len(tokens)-1] = tokens[:-1]
#     y[:len(tokens)-1] = tokens[1:]
#     ix[:len(tokens)-2] = tokens[1:-1]
    
#     return x, y, ix