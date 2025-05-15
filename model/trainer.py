
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from dataclasses import dataclass
from torch.optim.lr_scheduler import CosineAnnealingLR
from sentence_transformers import SentenceTransformer
from lightning.pytorch.utilities import grad_norm

class trainer(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.config = iGPTConfig()
        self.network = iGPT(self.config)
        self.lr = 6e-4
        self.weight_decay = 0.1 


    def training_step(self, batch, batch_idx):
        self.log('global_step', self.global_step)

        x, y, sentence_list = batch
        predicted_y = self.network(x, sentence_list)
        loss = F.cross_entropy(predicted_y.view(-1, predicted_y.size(-1)), y.view(-1))
        # Logging
        self.log("ahihi/Train Loss", loss, prog_bar=True)
        return loss
        

    def validation_step(self, batch, batch_idx):
        x, y, sentence_list = batch
        predicted_y = self.network(x, sentence_list)
        loss = F.cross_entropy(predicted_y.view(-1, predicted_y.size(-1)), y.view(-1))

        self.log("ahihi/Val loss", loss, prog_bar=True)
        return loss
        

    def test_step(self, batch, batch_idx):
        pass


    def forward(self, batch):
        x, ix = batch
        logits = self.network(x, ix)
        return logits
    
    
    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for (n, p) in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for (n, p) in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            # fused=use_fused
        )

        lr_scheduler = {'scheduler': CosineAnnealingLR(optimizer=optimizer, T_max=500, eta_min=self.lr*0.1),
                    'name': 'learning_rate',
                    'interval':'step',
                    'frequency': 1}
        
        return [optimizer], [lr_scheduler]

    def on_before_optimizer_step(self, optimizer):
        norm = grad_norm(self.network, norm_type=2)
        avg_norm = sum(norm.values())/len(norm)
        self.log('ahihi/norm', avg_norm, prog_bar=True)

            