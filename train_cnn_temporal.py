import os
import cv2
import numpy as np
import pandas as pd
from typing import Optional, TypedDict, Dict
import argparse

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchmetrics.classification import BinaryAccuracy
from sklearn.metrics import average_precision_score
from torchmetrics import Metric, MetricCollection
import wandb
from pytorch_lightning.loggers import WandbLogger
import pickle

from train_pl_v_swin_32frm import CollisionDataModule, TTEAPCollection

class TemporalAttention(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.channels = channels
        self.time_dim = time_dim
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Conv3d(channels, channels // 8, kernel_size=1),
            nn.BatchNorm3d(channels // 8),
            nn.ReLU(),
            nn.Conv3d(channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        attention = self.temporal_attention(x)  # (B, 1, T, H, W)
        return x * attention

class CollisionLitModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        # define a simple 3D CNN backbone and classification head
        self.backbone = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            nn.Conv3d(16, 32, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2))
        )
        self.temporal_attention = TemporalAttention(channels=32, time_dim=32)
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))  # only after attention
        self.classifier = nn.Linear(32, 1)
        self.lr = lr
        # loss and metrics
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        # custom validation metrics per TTE
        self.val_aps = TTEAPCollection([0.5, 1.0, 1.5], prefix="val_", compute_on_cpu=True)

    def forward(self, x):
        # x: (B, C, T, H, W)
        features = self.backbone(x)            # (B, 32, T, H/4, W/4)
        features = self.temporal_attention(features)  # Apply temporal attention
        features = self.pool(features)         # (B, 32, 1, 1, 1)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)     # (B, 1)
        return logits

    def training_step(self, batch, batch_idx):
        frames, labels, ttes = batch
        logits = self(frames)
        loss = self.criterion(logits.view(-1), labels)
        # log loss and accuracy
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        preds = torch.sigmoid(logits)
        self.train_acc(preds.view(-1), labels)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frames, labels, ttes = batch
        logits = self(frames)
        preds = torch.sigmoid(logits).view(-1)
        loss = self.criterion(logits.view(-1), labels)
        # log validation loss
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        # update custom AP metrics
        self.val_aps.update(preds, labels, ttes)
        self.log_dict(self.val_aps, on_step=False, on_epoch=True)
        # update accuracy
        self.val_acc(preds, labels)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main():
    # -------------- Parse arguments --------------
    parser = argparse.ArgumentParser(description='Train CNN model for collision detection')
    parser.add_argument('--sample_choice', type=str, default='end_biased',
                      choices=['uniform', 'end_biased', 'random', 'last_segment'],
                      help='Frame sampling strategy (default: end_biased)')
    args = parser.parse_args()

    # -------------- W&B --------------
    import os
    batch_limit_params = {"max_epochs": 10}
    if os.getenv("DO_DEBUG") == "1":
        wandb_logger = None
        batch_limit_params = {
            # "limit_train_batches": 5,
            "limit_val_batches": 5,
            "profiler": "simple",
            "max_epochs": 2,
        }
        device_num = 1
    else:
        # Define hyperparameters
        config = {
            "sample_choice": args.sample_choice,  # Use command line argument
            "batch_size": 16,
            "num_workers": 4,
            "learning_rate": 1e-4,
            "max_epochs": 30,
        }
        wandb_logger = WandbLogger(
            project="crash-cnn-32frm-temporal", 
            log_model=True,
            config=config,
            version=f"norm_sample_choice_{args.sample_choice}"
        )
        device_num = 1

    # -------------- datamodule --------------
    dm = CollisionDataModule(
        train_meta_path="data/split/train_clip_metadata_downsampling_32.json",
        val_meta_path="data/split/val_clip_metadata_downsampling_32.json",
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        transforms=None,
        sample_choice=config["sample_choice"],
    )

    # -------------- model --------------
    model = CollisionLitModel(lr=config["learning_rate"])

    # -------------- trainer --------------
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=device_num,
        strategy="auto",
        precision="bf16",
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                save_top_k=1,
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                mode="min",
                verbose=True
            )
        ],
        **batch_limit_params
    )

    # -------------- train --------------
    trainer.fit(model, dm)
    # trainer.save_checkpoint("collision_y.ckpt")
    wandb.finish()


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.backends.cudnn.benchmark = True
    main() 