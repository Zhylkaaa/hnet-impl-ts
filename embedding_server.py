import numpy as np
import torch
from torch import nested, Tensor as TT
from pytriton.triton import Triton
from pytriton.decorators import batch, sample
from pytriton.model_config import ModelConfig, Tensor

from hnet_impl import HNetConfig, HNetTS


def NJT(ls: list[TT]):
    return nested.nested_tensor(ls, layout=torch.jagged)

    
def impute(ecg: np.ndarray):
    ecg[np.isnan(ecg)] = 0
    return ecg


def normalize(ecg: np.ndarray):
    return (ecg - ecg.mean(axis=1, keepdims=True)) / (ecg.std(axis=1, keepdims=True) + 1e-6)


def process_ecg(ecg: np.ndarray):
    ecg = impute(ecg)
    ecg = normalize(ecg)
    return ecg


class EmbeddingModel:
    def __init__(self):
        ckpt = torch.load('ecg-ts/oit786xh/checkpoints/epoch=9-step=174840.ckpt', weights_only=False)
        config = ckpt['hyper_parameters']['config']
        use_simple_embedding = False
        with torch.device("cuda"):
            self.model = HNetTS(config, use_simple_embedding=use_simple_embedding)
        self.model.backbone.block_compile(ac=False)
        self.model.load_state_dict({k.replace('model.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}, strict=True)
        self.embedding_pulling = ckpt['hyper_parameters']['embedding_pulling']
    
    @batch
    @torch.inference_mode()
    def infer(self, inputs: np.ndarray):
        inputs = torch.from_numpy(process_ecg(inputs)).cuda()
        inputs = inputs.squeeze(-1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embeddings_flat, extras = self.model(NJT(inputs))
        embeddings_idx_flat = extras[0].b.values().reshape(inputs.shape).sum(dim=1)
        embeddings_idx = torch.cumsum(embeddings_idx_flat, dim=0)
        if self.embedding_pulling == 'last':
            embeddings = embeddings_flat.index_select(0, embeddings_idx - 1)
        elif self.embedding_pulling == 'mean':
            embeddings = torch.stack([t.mean(dim=0) for t in embeddings_flat.split(embeddings_idx_flat.tolist())])
        else:
            raise ValueError(f"Invalid embedding pulling method: {self.embedding_pulling}")
        return {"embeddings": embeddings.float().cpu().numpy()}


if __name__ == "__main__":
    with Triton() as triton:
        model = EmbeddingModel()
        triton.bind(
            model_name="EmbeddingModel",
            infer_func=model.infer,
            inputs=[
                Tensor(name="inputs",shape=(-1, 1), dtype=np.float32),
            ],
            outputs=[Tensor(name="embeddings", shape=(-1, -1), dtype=np.float32)],
            config=ModelConfig(batching=True, max_batch_size=64),
        )
        triton.serve()