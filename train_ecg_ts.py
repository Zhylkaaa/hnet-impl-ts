import os
from contextlib import nullcontext
from typing import Tuple

import torch
import torch.nn as nn
from torch import nested, Tensor as TT
from torch.optim.lr_scheduler import LRScheduler
import torch.distributed as dist
# from torch.distributed import device_mesh as tdm, fsdp

from hnet_impl import HNetConfig, HNetTS
from dataset import EGCDatamodule

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping


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
                 config: HNetConfig, 
                 model: nn.Module, 
                 lr: float, 
                 weight_decay: float, 
                 warmup_portion: float, 
                 embedding_pulling: str = 'mean', 
                 alpha: float = 0.03,
                 temperature: float = 0.07,
                 lr_scheduler: str = 'linear',
                 **kwargs):
        super().__init__()
        self.config = config
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: TT, input_mask: TT):
        return self.model(x, input_mask)
    
    def _common_step(self, batch: Tuple[TT, TT], stage: str):
        X, y, input_mask = batch
        zero = torch.tensor(0.0, device="cuda")
        with torch.autocast("cuda", torch.bfloat16):
            embeddings_flat, extras = self(X, input_mask=input_mask)
            embeddings_idx_flat = extras[0].b.values().reshape(input_mask.shape[0], X.shape[1]).sum(dim=1)
            embeddings_idx = torch.cumsum(embeddings_idx_flat, dim=0)
            if self.hparams.embedding_pulling == 'last':
                embeddings = embeddings_flat.index_select(0, embeddings_idx - 1)
            elif self.hparams.embedding_pulling == 'mean':
                embeddings = torch.stack([t.mean(dim=0) for t in embeddings_flat.split(embeddings_idx_flat.tolist())])
            else:
                raise ValueError(f"Invalid embedding pulling method: {self.hparams.embedding_pulling}")
            embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
            similarities = torch.matmul(embeddings_norm, embeddings_norm.transpose(0, 1))

            similarities[torch.arange(similarities.shape[0]), torch.arange(similarities.shape[0])] = -torch.inf
            similarities = similarities / self.hparams.temperature
            sim_loss = self.loss_fn(similarities, y)

            l_ratio = sum([e.loss_ratio for e in extras], zero)
            loss = sim_loss + self.hparams.alpha * l_ratio
            batch_size = input_mask.shape[0]
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
            self.log(f"{stage}_sim_loss", sim_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
            self.log(f"{stage}_l_ratio", l_ratio, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
            self.log(f"{stage}_sparsity", embeddings_idx_flat.float().mean().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
            return loss
    
    def training_step(self, batch: Tuple[TT, TT]):
        return self._common_step(batch, "train")
    
    # def validation_step(self, batch: Tuple[TT, TT]):
    #     return self._common_step(batch, "val")
    
    # def test_step(self, batch: Tuple[TT, TT]):
    #     return self._common_step(batch, "test")
    
    def configure_optimizers(self):
        base_lr = self.hparams.lr
        weight_decay = self.hparams.weight_decay
        warmup_portion = self.hparams.warmup_portion

        opt = torch.optim.AdamW(
            [
                dict(params=ls, lr=base_lr * lr_mod)
                for ls, lr_mod in zip(self.model.split_params_by_hierachy(), self.config.lambda_s())
            ],
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
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


if __name__ == "__main__":
    data_path = "/data/dz2449/ts_project/data/mimic_iv_ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
    seed = 42
    batch_size = 16
    gradient_accumulation_steps = 2
    num_workers = 16
    base_lr = 5e-4
    weight_decay = 0.01
    num_epochs = 10
    alpha = 0.03
    embedding_pulling = 'mean' # 'last' or 'mean'
    warmup_portion = 0.01
    temperature = 0.07
    layer_dims = [256, 512]
    arch = ["m4", "T6"]
    N_compress = [1, 10]
    lr_scheduler = 'linear'
    embedding_type = 'mamba_attention_multichannel'
    multichannel = True
    

    L.seed_everything(seed, workers=True)

    ## create model
    config = HNetConfig.create_reasonable_config_ts(D=layer_dims, arch=arch, N_compress=N_compress, embedding_type=embedding_type)
    print(config)
    with torch.device("cuda"):
        model = HNetTS(config)
    print(model)
    model.backbone.block_compile(ac=False)
    # if embedding_type != 'simple':
    #     model.embeddings.mamba_embedding = torch.compile(model.embeddings.mamba_embedding, fullgraph=True)
    # model.embeddings.cross_attn = torch.compile(model.embeddings.cross_attn, mode="reduce-overhead", fullgraph=True)
    if embedding_type == 'mamba_attention_multichannel':
        model.embeddings = torch.compile(model.embeddings, mode="reduce-overhead", dynamic=False, fullgraph=True)
    if embedding_type == 'gated_mamba_multichannel':
        model.embeddings.mamba_embedding = torch.compile(model.embeddings.mamba_embedding, mode="default", fullgraph=True)
        model.embeddings._aggregate_embeddings = torch.compile(model.embeddings._aggregate_embeddings, mode="reduce-overhead", dynamic=False, fullgraph=True)

    datamodule = EGCDatamodule(data_path, batch_size, num_workers, multichannel=multichannel)

    kwargs = {
        'lr': base_lr,
        'weight_decay': weight_decay,
        'alpha': alpha,
        'embedding_pulling': embedding_pulling, # 'last' or 'mean'
        'warmup_portion': warmup_portion,
        'temperature': temperature,
        'lr_scheduler': lr_scheduler,
    }
    model = ECGTSModel(config, model, **kwargs)

    loggers = [WandbLogger(project="ecg-ts")]

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
            **kwargs, arch=arch, layer_dims=layer_dims, N_compress=N_compress, 
            num_epochs=num_epochs, batch_size=batch_size, num_workers=num_workers, 
            gradient_accumulation_steps=gradient_accumulation_steps, 
            embedding_type=embedding_type,
            seed=seed,
            multichannel=multichannel,
        ))
    
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        accumulate_grad_batches=gradient_accumulation_steps,
        precision="bf16-mixed",
        max_epochs=num_epochs,
        enable_checkpointing=True,
        callbacks=[
            ModelCheckpoint(monitor="train_sim_loss_epoch", mode="min", save_top_k=1, save_last=True),
            LearningRateMonitor(),
        ],
        logger=loggers,
    )
    trainer.fit(model, datamodule=datamodule)
