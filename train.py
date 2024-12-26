import lightning as L
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
import os
from lightning.pytorch.callbacks import TQDMProgressBar
from utils.dataloaderlite import SentenceDataset, preprocessText, StreamingCsvDataset
from model.iGPT import NotMyModel

batch_size = 84
accumulation_step = 1

root_dir = "experiments" # log folder
resume_ckpt = None  # checkpoint path, 

data_root = 'data/wikitext'
shards = os.listdir('data/wikitext')
shards = [os.path.join(data_root, s) for s in shards]

def main():
    L.seed_everything(666)

    # Model
    model = NotMyModel()

    train_dataset = StreamingCsvDataset(file_paths=shards, 
                                        max_length=model.config.block_size, 
                                        text_column=0, 
                                        skip_header=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)

    # train_dataset = SentenceDataset(max_length=model.config.block_size)
    # train_dataset, val_dataset  = random_split(dataset, [n-200, 200])
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    
    # Trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2, # Save top k checkpoints
        monitor="global_step",
        mode="max",
    )
    trainer = L.Trainer(default_root_dir=root_dir,
                        callbacks=[checkpoint_callback, lr_monitor, TQDMProgressBar(refresh_rate=50)],
                        strategy='ddp_find_unused_parameters_true',
                        log_every_n_steps=1,
                        gradient_clip_val=1.0,
                        max_steps=5000,
                        accumulate_grad_batches= accumulation_step,
                        num_nodes=1)

    # Train
    trainer.fit(model, train_dataloader, ckpt_path=resume_ckpt)
    

if __name__ == "__main__":
    main()