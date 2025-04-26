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

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmaction.models import build_model
from mmaction.datasets.pipelines import Normalize

class TTEAP(Metric):
    def __init__(self, tte_value: float, **kwargs):
        super().__init__(**kwargs)
        self.tte_value = tte_value
        self.add_state("preds",  default=torch.tensor([], dtype=torch.float), dist_reduce_fx="cat")
        self.add_state("target", default=torch.tensor([], dtype=torch.float), dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor, ttes: torch.Tensor):
        mask = (ttes.squeeze() == self.tte_value) | torch.isnan(ttes.squeeze())
        # Only update if we have any matching samples
        preds = preds.view(-1)
        target = target.view(-1)
        if mask.any():
            self.preds = torch.cat([self.preds, preds[mask]])
            self.target = torch.cat([self.target, target[mask]])

    def compute(self):
        if self.preds.numel() == 0 or self.target.numel() == 0:
            return torch.tensor(0.0)
        y_true   = self.target.cpu().numpy()
        y_scores = self.preds.cpu().numpy()
        return torch.tensor(average_precision_score(y_true, y_scores), dtype=torch.float)

class TTEAPCollection(MetricCollection):
    def __init__(self, tte_values, prefix="", **kwargs):
        metrics = {f"ap@{tte}".replace('.', '_'): TTEAP(tte, **kwargs) for tte in tte_values}
        super().__init__(metrics, prefix=prefix)

class CollisionClipDataset(Dataset):
    def __init__(self, meta_data_path: str, sample_choice: str = "end_biased"):
        """
        CSV must have columns:
          video_path, target, time_of_event, time_of_alert
        """
        df = pd.read_json(meta_data_path)
        self.samples = []
        self.normalize = Normalize(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_bgr=False
        )
        for i, row in df.iterrows():
            if len(row["downsampling_choices"][sample_choice]) != 32:  # Changed to 32 for SlowFast
                print(f"Skipping {row['uid']} due to invalid downsampling choices")
                continue
            tensor_dict_path = f"data/split/processed_train/{row['uid']}_tensor_dict.pickle"
            sample = {
                "tensor_dict_path": tensor_dict_path,
                "label": row["label"],
                "frame_idxs": row["downsampling_choices"][sample_choice],
            }
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tensor_dict = pickle.load(open(sample["tensor_dict_path"], "rb"))
        frames_tensor = torch.stack([tensor_dict[i] for i in sample["frame_idxs"]])
        label = torch.tensor(sample["label"], dtype=torch.float)
        # T x C x H x W -> C, T, H, W
        frames_tensor = frames_tensor.permute(1, 0, 2, 3).to(torch.float)
        
        # Convert to numpy for MMAction2 normalization
        frames_np = frames_tensor.permute(1, 2, 3, 0).numpy()  # T, H, W, C
        results = dict(imgs=frames_np, modality='RGB')
        results = self.normalize(results)
        frames_tensor = torch.from_numpy(results['imgs']).permute(3, 0, 1, 2)  # C, T, H, W
        
        return frames_tensor, label, 0

class CollisionValidationClipDataset(Dataset):
    def __init__(self, meta_data_path: str, sample_choice: str = "end_biased"):
        df = pd.read_json(meta_data_path)
        self.samples = []
        self.normalize = Normalize(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_bgr=False
        )
        for i, row in df.iterrows():
            if len(row["downsampling_choices"][sample_choice]) != 32:  # Changed to 32 for SlowFast
                print(f"Skipping {row['uid']} due to invalid downsampling choices")
                continue
            tensor_dict_path = f"data/split/processed_val/{row['uid']}_tensor_dict.pickle"
            sample = {
                "tensor_dict_path": tensor_dict_path,
                "label": row["label"],
                "TTE": row["ttc"],
                "normal_idxs": row["downsampling_choices"][sample_choice],
            }
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if sample['label'] == 1:
            assert sample["TTE"] in [0.5, 1.0, 1.5], f"label: {sample['label']}, TTE: {sample['TTE']}"
        else:
            assert pd.isna(sample["TTE"]), f"label: {sample['label']}, TTE: {sample['TTE']}"
        tensor_dict = pickle.load(open(sample["tensor_dict_path"], "rb"))
        frames_tensor = torch.stack([tensor_dict[i] for i in sample["normal_idxs"]])
        label = torch.tensor(sample["label"], dtype=torch.float)
        # T x C x H x W -> C, T, H, W
        frames_tensor = frames_tensor.permute(1, 0, 2, 3).to(torch.float)
        
        # Convert to numpy for MMAction2 normalization
        frames_np = frames_tensor.permute(1, 2, 3, 0).numpy()  # T, H, W, C
        results = dict(imgs=frames_np, modality='RGB')
        results = self.normalize(results)
        frames_tensor = torch.from_numpy(results['imgs']).permute(3, 0, 1, 2)  # C, T, H, W
        
        ttes = torch.tensor([sample["TTE"]], dtype=torch.float)
        
        return frames_tensor, label, ttes

class CollisionTestClipDataset(Dataset):
    def __init__(self, meta_data_path: str="data/test_clip_metadata_downsampling_32_frame.json", sample_choice: str = "end_biased"):
        df = pd.read_json(meta_data_path)
        self.samples = []
        self.normalize = Normalize(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_bgr=False
        )
        for i, row in df.iterrows():
            self.samples.append({
                "test_id": row["test_id"],
                "tensor_dict_path": f"data/split/processed_test/{row['test_id']}_tensor_dict.pickle",
                "frame_idxs": row["downsampling_choices"][sample_choice],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tensor_dict = pickle.load(open(sample["tensor_dict_path"], "rb"))
        frames_tensor = torch.stack([tensor_dict[i] for i in sample["frame_idxs"]])
        # T x C x H x W -> C, T, H, W
        frames_tensor = frames_tensor.permute(1, 0, 2, 3).to(torch.float)
        
        # Convert to numpy for MMAction2 normalization
        frames_np = frames_tensor.permute(1, 2, 3, 0).numpy()  # T, H, W, C
        results = dict(imgs=frames_np, modality='RGB')
        results = self.normalize(results)
        frames_tensor = torch.from_numpy(results['imgs']).permute(3, 0, 1, 2)  # C, T, H, W
        
        return frames_tensor, sample["test_id"]

class CollisionDataModule(pl.LightningDataModule):
    def __init__(self, train_meta_path: str, val_meta_path: str, batch_size: int = 8, num_workers: int = 4, transforms=None, sample_choice: str = "end_biased"):
        super().__init__()
        self.train_meta_path = train_meta_path
        self.val_meta_path = val_meta_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.sample_choice = sample_choice

    def setup(self, stage: Optional[str] = None):
        if stage == "predict":
            self.predict_dataset = CollisionTestClipDataset(getattr(self, "test_meta_path", "data/test_clip_metadata_downsampling_32_frame.json"), self.sample_choice)
        else:
            self.train_dataset = CollisionClipDataset(self.train_meta_path, self.sample_choice)
            self.val_dataset = CollisionValidationClipDataset(self.val_meta_path, self.sample_choice)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

class SlowFast(nn.Module):
    def __init__(self, num_classes: int = 1, freeze: bool = True):
        super().__init__()
        cfg = Config.fromfile(
            "configs/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py"
        )
        self.model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))

        load_checkpoint(
            self.model,
            "checkpoints/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb_20200728-145f1097.pth",
            strict=False,
        )

        if freeze:
            for p in self.model.backbone.parameters():
                p.requires_grad = False

        in_f = self.model.cls_head.fc_cls.in_features
        self.model.cls_head.fc_cls = nn.Linear(in_f, num_classes)

    def forward(self, x):
        features = self.model.extract_feat(x)
        return self.model.cls_head(features)

class CollisionLitModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, freeze: bool = True):
        super().__init__()
        self.net = SlowFast(num_classes=1, freeze=freeze)
        self.lr = lr
        # loss and metrics
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        # custom validation metrics per TTE
        self.val_aps = TTEAPCollection([0.5, 1.0, 1.5], prefix="val_", compute_on_cpu=True)

    def forward(self, x):
        return self.net(x).squeeze(1)

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

    def predict_step(self, batch, batch_idx):
        frames, test_ids = batch
        logits = self(frames)
        preds = torch.sigmoid(logits).view(-1)
        return {"test_ids": test_ids, "preds": preds}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def test_model(checkpoint_path: str, sample_choice: str = "end_biased"):
    # -------------- model --------------
    model = CollisionLitModel.load_from_checkpoint(checkpoint_path)
    
    # -------------- datamodule --------------
    dm = CollisionDataModule(
        train_meta_path="data/split/train_clip_metadata_downsampling_32.json",
        val_meta_path="data/split/val_clip_metadata_downsampling_32.json",
        batch_size=16,
        num_workers=8,
        transforms=None,
        sample_choice=sample_choice,
    )
    
    # -------------- trainer --------------
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision="32-true",
    )
    torch.set_float32_matmul_precision("high")
    
    # -------------- predict --------------
    predictions = trainer.predict(model, dm)
    
    # Combine predictions from all batches
    all_test_ids = []
    all_preds = []
    for batch_pred in predictions:
        all_test_ids.extend(batch_pred["test_ids"].cpu().numpy())
        all_preds.extend(batch_pred["preds"].cpu().numpy())
    
    # Create submission DataFrame with 5-digit IDs and 4-decimal scores
    submission_df = pd.DataFrame({
        "id": [f"{int(id):05d}" for id in all_test_ids],
        "score": [f"{score:.4f}" for score in all_preds]
    })
    
    # Sort by test_id to ensure consistent order
    submission_df = submission_df.sort_values("id")
    
    # Save to CSV with timestamp if file exists
    output_path = f"submission_slowfast_{sample_choice}.csv"
    if os.path.exists(output_path):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"submission_{sample_choice}_{timestamp}.csv"
    
    submission_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

def main():
    # -------------- Argument Parser --------------
    parser = argparse.ArgumentParser(description='Train SlowFast model for collision detection')
    parser.add_argument('--sample_choice', type=str, default='end_biased',
                      choices=['uniform', 'end_biased', 'last_segment', 'random'],
                      help='Sampling strategy for frames')
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--test_checkpoint_path", type=str, help="Path to checkpoint to test from")
    parser.add_argument("--freeze", action="store_true", help="Whether to freeze the backbone")
    args = parser.parse_args()

    if args.do_test:
        assert args.test_checkpoint_path is not None, "Must provide checkpoint path for testing"
        test_model(args.test_checkpoint_path, args.sample_choice)
        return

    # -------------- W&B --------------
    import os
    batch_limit_params = {"max_epochs": 30}
    if os.getenv("DO_DEBUG") == "1":
        wandb_logger = None
        batch_limit_params = {
            "limit_val_batches": 5,
            "profiler": "simple",
            "max_epochs": 2,
        }
        device_num = 1
    else:
        # Define hyperparameters
        config = {
            "sample_choice": args.sample_choice,
            "batch_size": 16,
            "num_workers": 8,
            "learning_rate": 1e-5,
            "max_epochs": 30,
            "freeze": args.freeze,
        }
        version = f"trial_{config['sample_choice']}"
        if args.freeze:
            version += "_frozen"
        else:
            version += "_full_finetune"
        wandb_logger = WandbLogger(
            project="crash-slowfast",
            log_model=True,
            config=config,
            version=version,
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
    model = CollisionLitModel(lr=config["learning_rate"], freeze=config["freeze"])

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
            )
        ],
        **batch_limit_params
    )

    # -------------- train --------------
    if args.resume:
        trainer.fit(model, dm, ckpt_path=args.resume)
    else:
        trainer.fit(model, dm)
    wandb.finish()

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.backends.cudnn.benchmark = True
    main() 
