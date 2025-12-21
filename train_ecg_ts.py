import os
from contextlib import nullcontext
from typing import Tuple

import torch
import torch.nn as nn
from torch import nested, Tensor as TT
from torch.optim.lr_scheduler import LRScheduler
import torch.distributed as dist
from torch.distributed import device_mesh as tdm, fsdp

from hnet_impl import HNetConfig, HNetTS
from dataset import EGCDatamodule

import wandb
# import lightning as L
# from lightning.pytorch.loggers import WandbLogger
# from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping


def NJT(ls: list[TT]):
        return nested.nested_tensor(ls, layout=torch.jagged)


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


# class ECGTSModel(L.LightningModule):
#     def __init__(self, config: HNetConfig, model: nn.Module, lr: float, weight_decay: float):
#         super().__init__()
#         self.config = config
#         self.model = model
#         self.save_hyperparameters(ignore=["model"])
#         self.loss_fn = nn.CrossEntropyLoss()

#     def forward(self, x: TT):
#         return self.model(x)
    
#     def _common_step(self, batch: Tuple[TT, TT], stage: str):
#         X, y = batch
#         with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#             embeddings_flat, extras = self(NJT(X))
#         embeddings_idx = extras[0].b.values().reshape(X.shape).sum(dim=1)
#         embeddings_idx = torch.cumsum(embeddings_idx, dim=0)
#         embeddings = embeddings_flat.index_select(0, embeddings_idx)
#         embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
#         similarities = torch.matmul(embeddings_norm, embeddings_norm.transpose(0, 1))
#         # zero out the diagonal
#         similarities[torch.arange(similarities.shape[0]), torch.arange(similarities.shape[0])] = 0.
#         loss = self.loss_fn(similarities, y)
#         self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss
    
#     def training_step(self, batch: Tuple[TT, TT]):
#         return self._common_step(batch, "train")
    
#     def validation_step(self, batch: Tuple[TT, TT]):
#         return self._common_step(batch, "val")
    
#     def test_step(self, batch: Tuple[TT, TT]):
#         return self._common_step(batch, "test")
    
#     def configure_optimizers(self):
#         base_lr = self.hparams.lr
#         weight_decay = self.hparams.weight_decay

#         opt = torch.optim.AdamW(
#             [
#                 dict(params=ls, lr=base_lr * lr_mod)
#                 for ls, lr_mod in zip(self.model.split_params_by_hierachy(), self.config.lambda_s())
#             ],
#             betas=(0.9, 0.95),
#             weight_decay=weight_decay,
#         )
#         num_training_steps = self.trainer.estimated_stepping_batches
#         num_warmup_steps = num_training_steps * 0.05
#         lr_scheduler = LinearWarmupDecayLR(opt, num_warmup_steps, num_training_steps)
#         return {
#                 "optimizer": opt,
#                 "lr_scheduler": {
#                     "scheduler": lr_scheduler,
#                     "monitor": 'val_loss_epoch',
#                     "interval": 'step',
#                     "frequency": 1,
#                     "strict": True,
#                 },
#             }

def is_main_process():
    # Check if distributed training is enabled and if the current rank is 0
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


if __name__ == "__main__":
    data_path = "/data/dz2449/ts_project/data/mimic_iv_ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
    batch_size = 16
    num_workers = 16
    base_lr = 1e-3
    weight_decay = 0.01
    num_epochs = 10
    alpha = 0.03
    embedding_pulling = 'mean' # 'last' or 'mean'
    warmup_portion = 0.01

    ws = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mesh = tdm.init_device_mesh("cuda", (ws,), mesh_dim_names=("dp",))
    ## create model
    config = HNetConfig.create_reasonable_config_ts(D=[256, 512], arch=["m4", "T6"], N_compress=[1, 10])
    print(config)
    with torch.device("cuda"):
        model = HNetTS(config)
    print(model)
    model.backbone.block_compile(ac=False)
    model.apply_fsdp(  # default: BF16, ZeRO2, 1D mesh
        model,
        mp_policy=fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        reshard_after_forward=False,
        mesh=mesh["dp"],
    ) if ws > 1 else model

    opt = torch.optim.AdamW(
        [
            dict(params=ls, lr=base_lr * lr_mod)
            for ls, lr_mod in zip(model.split_params_by_hierachy(), config.lambda_s())
        ],
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    datamodule = EGCDatamodule(data_path, batch_size, num_workers)

    num_training_steps = len(datamodule.train_dataset) // batch_size * num_epochs
    num_warmup_steps = num_training_steps * warmup_portion
    print(f"Num warmup steps: {num_warmup_steps}, Num training steps: {num_training_steps}")
    lr_scheduler = LinearWarmupDecayLR(opt, num_warmup_steps, num_training_steps)

    exclude_extensions = (".yaml", ".gz", ".pt", ".tar", ".csv")
    if is_main_process():
        run = wandb.init(project="hnet-impl")
        run.log_code(
            ".", include_fn=lambda path: path.endswith(".py"), exclude_fn=lambda path: path.endswith(exclude_extensions) or '.venv' in path or 'wandb' in path
        )
    else:
        # For non-main processes, initialize W&B in 'disabled' mode
        run = wandb.init(mode="disabled")
    
    
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        try:
            for step, batch in enumerate(datamodule.train_dataloader()):
                X, y = batch
                X = X.cuda()
                y = y.cuda()
                zero = torch.tensor(0.0, device="cuda")
                with torch.autocast("cuda", torch.bfloat16):
                    embeddings_flat, extras = model(NJT(X))
                    embeddings_idx_flat = extras[0].b.values().reshape(X.shape).sum(dim=1)
                    embeddings_idx = torch.cumsum(embeddings_idx_flat, dim=0)
                    if embedding_pulling == 'last':
                        embeddings = embeddings_flat.index_select(0, embeddings_idx - 1)
                    elif embedding_pulling == 'mean':
                        embeddings = torch.stack([t.mean(dim=0) for t in embeddings_flat.split(embeddings_idx_flat.tolist())])
                    else:
                        raise ValueError(f"Invalid embedding pulling method: {embedding_pulling}")
                    embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    similarities = torch.matmul(embeddings_norm, embeddings_norm.transpose(0, 1))

                    similarities[torch.arange(similarities.shape[0]), torch.arange(similarities.shape[0])] = 0.
                    sim_loss = loss_fn(similarities, y)

                    l_ratio = sum([e.loss_ratio for e in extras], zero)
                    loss = sim_loss + alpha * l_ratio
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    lr_scheduler.step()
                    # print(X)
                    # print(y)
                    # print(X.sum(dim=1))
                    # break
                    run.log({"train_loss": loss.item(), "sim_loss": sim_loss.item(), "l_ratio": l_ratio.item(), "sparsity": embeddings_idx_flat.float().mean().item()})
                    if step % 100 == 0:
                        print(f"Epoch {epoch}, Step {step}, Loss {loss.item()}")
        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            print(X)
            print(y)
            print(X.sum(dim=1))
            raise