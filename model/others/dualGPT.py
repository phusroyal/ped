import torch
import torch.nn as nn
import inspect

class DualGPT(nn.Module):
    """
    Wraps both main GPT (decoder-only) and iGPT (aux model).
    Provides a single interface for forward, and a configure_optimizers method.
    """

    def __init__(self, main_gpt, idea_gpt):
        super().__init__()
        self.main_gpt = main_gpt
        self.idea_gpt = idea_gpt

    def forward(self, x=None, y=None, ix=None):
        device = x.device
        pred_idea = self.idea_gpt(ix).to(device)
        logits, loss = self.main_gpt(x, targets=y, idea_vector=pred_idea)
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device, master_process=True):
        """
        Example usage:
          optimizer = dual_gpt.configure_optimizers(
              weight_decay=0.1, 
              learning_rate=3e-4, 
              device=device,
              master_process=(ddp_rank==0)
          )
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for (n, p) in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for (n, p) in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, 
            lr=learning_rate, 
            betas=(0.9, 0.95), 
            eps=1e-8, 
            fused=use_fused
        )
        return optimizer