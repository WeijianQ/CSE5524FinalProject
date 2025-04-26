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

# class TTEAP(Metric):
#     def __init__(self, tte_value: float, **kwargs):
#         super().__init__(**kwargs)
#         self.tte_value = tte_value
#         self.add_state("preds",  default=torch.tensor([], dtype=torch.float), dist_reduce_fx="cat")
#         self.add_state("target", default=torch.tensor([], dtype=torch.float), dist_reduce_fx="cat")

#     def update(self, preds: torch.Tensor, target: torch.Tensor, ttes: torch.Tensor):
#         mask = (ttes.squeeze() == self.tte_value) | torch.isnan(ttes.squeeze())
#         # Only update if we have any matching samples
#         preds = preds.view(-1)
#         target = target.view(-1)
#         if mask.any():
#             self.preds = torch.cat([self.preds, preds[mask]])
#             self.target = torch.cat([self.target, target[mask]])

#     def compute(self):
#         if self.preds.numel() == 0 or self.target.numel() == 0:
#             return torch.tensor(0.0)
#         y_true   = self.target.cpu().numpy()
#         y_scores = self.preds.cpu().numpy()
#         return torch.tensor(average_precision_score(y_true, y_scores), dtype=torch.float)

# class TTEAPCollection(MetricCollection):
#     def __init__(self, tte_values, prefix="", **kwargs):
#         metrics = {f"ap@{tte}".replace('.', '_'): TTEAP(tte, **kwargs) for tte in tte_values}
#         super().__init__(metrics, prefix=prefix)

# class CollisionClipDataset(Dataset):
#     def __init__(self, meta_data_path: str, sample_choice: str = "end_biased"):
#         """
#         CSV must have columns:
#           video_path, target, time_of_event, time_of_alert
#         """
#         df = pd.read_json(meta_data_path)
#         self.samples = []
#         self.normalize = Normalize(
#             mean=[123.675, 116.28, 103.53],
#             std=[58.395, 57.12, 57.375],
#             to_bgr=False
#         )
#         for i, row in df.iterrows():
#             # data/split/processed_train/clip_0_1p0_tensor_dict.pickle
#             if len(row["downsampling_choices"][sample_choice]) != 32:
#                 print(f"Skipping {row['uid']} due to invalid downsampling choices")
#                 continue
#             tensor_dict_path = f"data/split/processed_train/{row['uid']}_tensor_dict.pickle"
#             sample = {
#                 "tensor_dict_path": tensor_dict_path,
#                 "label": row["label"],
#                 "frame_idxs": row["downsampling_choices"][sample_choice],
#             }
#             self.samples.append(sample)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         tensor_dict = pickle.load(open(sample["tensor_dict_path"], "rb"))
#         frames_tensor = torch.stack([tensor_dict[i] for i in sample["frame_idxs"]])
#         # Convert to numpy for normalization
#         frames_np = frames_tensor.numpy()
#         results = dict(imgs=frames_np, modality='RGB')
#         results = self.normalize(results)
#         frames_tensor = torch.from_numpy(results['imgs']).permute(3, 0, 1, 2)  # C, T, H, W
#         label = torch.tensor(sample["label"], dtype=torch.float)
#         return frames_tensor, label, 0

# class CollisionValidationClipDataset(Dataset):
#     def __init__(self, meta_data_path: str, sample_choice: str = "end_biased"):
#         df = pd.read_json(meta_data_path)
#         self.samples = []
#         self.normalize = Normalize(
#             mean=[123.675, 116.28, 103.53],
#             std=[58.395, 57.12, 57.375],
#             to_bgr=False
#         )
#         for i, row in df.iterrows():
#             # data/split/processed_val/clip_0_1p0_tensor_dict.pickle
#             if len(row["downsampling_choices"][sample_choice]) != 32:
#                 print(f"Skipping {row['uid']} due to invalid downsampling choices")
#                 continue
#             tensor_dict_path = f"data/split/processed_val/{row['uid']}_tensor_dict.pickle"
#             sample = {
#                 "tensor_dict_path": tensor_dict_path,
#                 "label": row["label"],
#                 "TTE": row["ttc"],
#                 "normal_idxs": row["downsampling_choices"][sample_choice],
#             }
#             self.samples.append(sample)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         if sample['label'] == 1:
#             assert sample["TTE"] in [0.5, 1.0, 1.5], f"label: {sample['label']}, TTE: {sample['TTE']}"
#         else:
#             assert pd.isna(sample["TTE"]), f"label: {sample['label']}, TTE: {sample['TTE']}"
#         tensor_dict = pickle.load(open(sample["tensor_dict_path"], "rb"))
#         frames_tensor = torch.stack([tensor_dict[i] for i in sample["normal_idxs"]])
#         # Convert to numpy for normalization
#         frames_np = frames_tensor.numpy()
#         results = dict(imgs=frames_np, modality='RGB')
#         results = self.normalize(results)
#         frames_tensor = torch.from_numpy(results['imgs']).permute(3, 0, 1, 2)  # C, T, H, W
#         label = torch.tensor(sample["label"], dtype=torch.float)
#         ttes = torch.tensor([sample["TTE"]], dtype=torch.float)
        
#         return frames_tensor, label, ttes

# class CollisionDataModule(pl.LightningDataModule):
#     def __init__(self, train_meta_path: str, val_meta_path: str, batch_size: int = 8, num_workers: int = 4, transforms=None, sample_choice: str = "end_biased"):
#         super().__init__()
#         self.train_meta_path = train_meta_path
#         self.val_meta_path = val_meta_path
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.transforms = transforms
#         self.sample_choice = sample_choice

#     def setup(self, stage: Optional[str] = None):
#         self.train_dataset = CollisionClipDataset(self.train_meta_path, self.sample_choice)
#         self.val_dataset = CollisionValidationClipDataset(self.val_meta_path, self.sample_choice)

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers
#         )

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
            nn.MaxPool3d((1,2,2)),
            nn.AdaptiveAvgPool3d((1,1,1)),
        )
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
        features = self.backbone(x)            # (B, 32, 1, 1, 1)
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
            project="crash-cnn-32frm", 
            log_model=True,
            config=config,
            version=f"temporal_norm_sample_choice_{args.sample_choice}"
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