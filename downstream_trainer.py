import os
import argparse
from contextlib import nullcontext
from typing import Tuple

import torch
import torch.nn as nn
from torch import nested, Tensor as TT
from torch.optim.lr_scheduler import LRScheduler
import torch.distributed as dist
# from torch.distributed import device_mesh as tdm, fsdp

from hnet_impl import HNetConfig, HNetTS
from dataset import PTBXLEGCDatamodule

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torchmetrics.classification import MultilabelAUROC, MultilabelAccuracy


class LinearWarmupDecayLR(LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.num_warmup_steps:
            # Linear warmup
            scale = step / max(1, self.num_warmup_steps)
        else:
            # Linear decay
            scale = max(
                0.0,
                float(self.num_training_steps - step) / float(max(1, self.num_training_steps - self.num_warmup_steps)),
            )

        return [base_lr * scale for base_lr in self.base_lrs]


class ECGTSModel(L.LightningModule):
    def __init__(self, 
                 embedding_model: nn.Module,
                 embedding_dim: int,
                 num_classes: int,
                 lr: float, 
                 weight_decay: float, 
                 warmup_portion: float, 
                 embedding_pulling: str = 'mean', 
                 lr_scheduler: str = 'linear',
                 **kwargs):
        super().__init__()
        self.embedding_model = embedding_model
        self.embedding_model.requires_grad_(False)
        self.embedding_dim = embedding_dim
        self.projection_head = nn.Linear(embedding_dim, num_classes)
        self.embedding_pulling = embedding_pulling
        self.save_hyperparameters(ignore=["embedding_model"])
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Initialize metrics for each stage
        self.train_metrics = {
            'auroc': MultilabelAUROC(num_labels=num_classes),
            'accuracy': MultilabelAccuracy(num_labels=num_classes),
        }
        self.val_metrics = {
            'auroc': MultilabelAUROC(num_labels=num_classes),
            'accuracy': MultilabelAccuracy(num_labels=num_classes),
        }
        self.test_metrics = {
            'auroc': MultilabelAUROC(num_labels=num_classes),
            'accuracy': MultilabelAccuracy(num_labels=num_classes),
        }
        self.metrics = {
            'train': self.train_metrics,
            'val': self.val_metrics,
            'test': self.test_metrics,
        }
        for stage, metrics in self.metrics.items():
            for name, metric in metrics.items():
                self.register_module(f"{stage}_{name}", metric)

    def forward(self, inputs: TT):
        with torch.inference_mode():
            num_examples, num_channels, T, _ = inputs.shape
            inputs = inputs.reshape(num_examples * num_channels, T, 1)
            input_mask = torch.zeros(num_examples, num_channels, dtype=torch.bool, device=inputs.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                embeddings_flat, extras = self.embedding_model(inputs, input_mask)
            embeddings_idx_flat = extras[0].b.values().reshape(num_examples, T).sum(dim=1)
            embeddings_idx = torch.cumsum(embeddings_idx_flat, dim=0)
            if self.embedding_pulling == 'last':
                embeddings = embeddings_flat.index_select(0, embeddings_idx - 1)
            elif self.embedding_pulling == 'mean':
                embeddings = torch.stack([t.mean(dim=0) for t in embeddings_flat.split(embeddings_idx_flat.tolist())])
        return self.projection_head(embeddings.clone())
    
    def _common_step(self, batch: Tuple[TT, TT], stage: str):
        X, y = batch
        y_hat = self(X)
        loss = self.loss_fn(y_hat, y)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=X.shape[0])
        
        # Convert logits to probabilities for metrics
        y_probs = torch.sigmoid(y_hat)
        
        # Update metrics based on stage
        for metric_name, metric in self.metrics[stage].items():
            metric(y_probs, y.long())
            self.log(f"{stage}_{metric_name}", metric, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def training_step(self, batch: Tuple[TT, TT]):
        return self._common_step(batch, "train")
    
    def validation_step(self, batch: Tuple[TT, TT]):
        return self._common_step(batch, "val")
    
    def test_step(self, batch: Tuple[TT, TT]):
        return self._common_step(batch, "test")
    
    def configure_optimizers(self):
        base_lr = self.hparams.lr
        weight_decay = self.hparams.weight_decay
        warmup_portion = self.hparams.warmup_portion

        # lambda_s[1] = (lambda_s[1] ** 2 * 1.6) ** 0.5
        opt = torch.optim.AdamW(
            self.projection_head.parameters(),
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
            lr=base_lr,
        )
        scheduler_kwargs = {}
        if self.hparams.lr_scheduler == 'linear':
            num_training_steps = self.trainer.estimated_stepping_batches
            num_warmup_steps = num_training_steps * warmup_portion
            lr_scheduler = LinearWarmupDecayLR(opt, num_warmup_steps, num_training_steps)
            scheduler_kwargs = {
                "interval": 'step',
            }
        elif self.hparams.lr_scheduler == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-6)
            scheduler_kwargs = {
                'interval': 'epoch',
            }
        else:
            raise ValueError(f"Invalid lr scheduler: {self.hparams.lr_scheduler}")
        return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    **scheduler_kwargs
                },
            }

def is_main_process():
    # Check if distributed training is enabled and if the current rank is 0
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


SUBSETS = {
    'form': 19,
    'rhythm': 12,
    'diagnostic': 44,
    'subdiagnostic': 23,
    'supdiagnostic': 5
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ECG Time Series Model")
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="Path to the pretrained model")
    parser.add_argument("--subset", type=str, required=True, choices=['form', 'rhythm', 'diagnostic', 'subdiagnostic', 'supdiagnostic'],
                        help="Subset to train on")
    parser.add_argument("--data_path", type=str, 
                        default="/data/dz2449/ts_project/data/ptb-xl/ptb-xl/1.0.3",
                        help="Path to the dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Number of gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of data loader workers")
    parser.add_argument("--base_lr", type=float, default=1e-3, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--embedding_pulling", type=str, default='mean', choices=['last', 'mean'],
                        help="Embedding pulling method: 'last' or 'mean'")
    parser.add_argument("--warmup_portion", type=float, default=0.01, 
                        help="Portion of training steps for warmup")
    parser.add_argument("--lr_scheduler", type=str, default='linear', choices=['linear', 'cosine'],
                        help="Learning rate scheduler type")
    args = parser.parse_args()
    
    data_path = args.data_path
    seed = args.seed
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_workers = args.num_workers
    base_lr = args.base_lr
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    embedding_pulling = args.embedding_pulling
    warmup_portion = args.warmup_portion
    lr_scheduler = args.lr_scheduler
    pretrained_model_path = args.pretrained_model_path
    L.seed_everything(seed, workers=True)

    ## create model
    ckpt = torch.load(pretrained_model_path, weights_only=False)
    config = ckpt['hyper_parameters']['config']
    with torch.device("cuda"):
        model = HNetTS(config)
    model.backbone.block_compile(ac=False)
    print(config)
    with torch.device("cuda"):
        model = HNetTS(config)
    print(model)
    model.backbone.block_compile(ac=False)

    embedding_type = config.embedding_type

    if embedding_type == 'mamba_attention_multichannel':
        model.embeddings = torch.compile(model.embeddings, mode="reduce-overhead", dynamic=False, fullgraph=True)
    if embedding_type == 'gated_mamba_multichannel':
        model.embeddings.mamba_embedding = torch.compile(model.embeddings.mamba_embedding, mode="default", fullgraph=True)
        model.embeddings._aggregate_embeddings = torch.compile(model.embeddings._aggregate_embeddings, mode="reduce-overhead", dynamic=False, fullgraph=True)
    
    model.load_state_dict({k.replace('model.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}, strict=True)

    datamodule = PTBXLEGCDatamodule(data_path, args.subset, batch_size, num_workers, use_lead='all' if embedding_type.endswith('multichannel') else 'II')

    kwargs = {
        'lr': base_lr,
        'weight_decay': weight_decay,
        'embedding_pulling': embedding_pulling, # 'last' or 'mean'
        'warmup_portion': warmup_portion,
        'lr_scheduler': lr_scheduler,
    }
    embedding_dim = config.d_model[0]
    model = ECGTSModel(model, embedding_dim, SUBSETS[args.subset], **kwargs)

    loggers = [WandbLogger(project="ecg-ts-ptbxl")]

    exclude_extensions = (".yaml", ".gz", ".pt", ".tar", ".csv")
    def should_exclude(path: str) -> bool:
        """Return True if path should be excluded from code upload."""
        path_str = str(path)
        # Exclude by extension
        if path_str.endswith(exclude_extensions):
            return True
        # Exclude directories
        if '.venv' in path_str or 'wandb' in path_str or '__pycache__' in path_str:
            return True
        return False
    
    def should_include(path: str) -> bool:
        """Return True if path should be included in code upload."""
        return path.endswith(".py")
    
    print(loggers[0].experiment.log_code(
        include_fn=should_include,
        exclude_fn=should_exclude
    ))

    for logger in loggers:
        logger.log_hyperparams(dict(
            **kwargs,
            num_epochs=num_epochs, batch_size=batch_size, num_workers=num_workers, 
            gradient_accumulation_steps=gradient_accumulation_steps, 
            embedding_type=embedding_type,
            seed=seed,
            pretrained_model_path=pretrained_model_path,
            subset=args.subset,
        ))
    
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        accumulate_grad_batches=gradient_accumulation_steps,
        precision="bf16-mixed",
        max_epochs=num_epochs,
        enable_checkpointing=True,
        callbacks=[
            ModelCheckpoint(monitor="val_loss_epoch", mode="min", save_top_k=1, save_last=True),
            LearningRateMonitor(),
        ],
        logger=loggers,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")
