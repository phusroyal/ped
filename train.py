import lightning as L
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
import os
import glob
from lightning.pytorch.callbacks import TQDMProgressBar
from utils.dataloaderlite import SentenceDataset, preprocessText, StreamingCsvDataset, ValCsvDataset
from model.iGPT import NotMyModel

batch_size = 136
accumulation_step = 10

root_dir = "experiments" # log folder
resume_ckpt = None  # checkpoint path, 

data_root = 'data/wikitext'
train_shards = glob.glob(os.path.join(data_root, '*_train_*'))
val_shard = glob.glob(os.path.join(data_root, '*_val_*'))

def main():
    L.seed_everything(666)

    torch.set_float32_matmul_precision('high')

    # Model
    model = NotMyModel()

    train_dataset = StreamingCsvDataset(file_paths=train_shards, 
                                        max_length=model.config.block_size, 
                                        text_column=0, 
                                        skip_header=True)
    val_dataset = ValCsvDataset(file_path=val_shard[0],
                                max_length=model.config.block_size, 
                                text_column=0, 
                                skip_header=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    # Trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2, # Save top k checkpoints
        monitor="global_step",
        mode="max",
    )
    trainer = L.Trainer(default_root_dir=root_dir,
                        callbacks=[checkpoint_callback, lr_monitor, TQDMProgressBar(refresh_rate=50)],
                        strategy= 'ddp', #'ddp_find_unused_parameters_true',
                        log_every_n_steps=1,
                        gradient_clip_val=1.0,
                        max_epochs=50,
                        accumulate_grad_batches= accumulation_step,
                        num_nodes=1,
                        val_check_interval=100,
                        precision="bf16-mixed") # bf16-mixed, 16-mixed

    # Train
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=resume_ckpt)
    

if __name__ == "__main__":
    main()