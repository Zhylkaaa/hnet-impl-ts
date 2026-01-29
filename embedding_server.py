import numpy as np
import torch
from torch import nested, Tensor as TT
from pytriton.triton import Triton, TritonConfig
from pytriton.decorators import batch, sample
from pytriton.model_config import ModelConfig, Tensor, DynamicBatcher

from hnet_impl import HNetConfig, HNetTS
torch._dynamo.config.cache_size_limit = 32
torch._dynamo.config.recompile_limit = 128

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
        ckpt = torch.load('ecg-ts/0mittj80/checkpoints/epoch=9-step=250020.ckpt', weights_only=False)
        config = ckpt['hyper_parameters']['config']
        with torch.device("cuda"):
            self.model = HNetTS(config)
        self.model.backbone.block_compile(ac=False)
        embedding_type = config.embedding_type
        if embedding_type == 'mamba_attention_multichannel':
            self.model.embeddings = torch.compile(self.model.embeddings, mode="reduce-overhead", dynamic=False, fullgraph=True)
        if embedding_type == 'gated_mamba_multichannel':
            self.model.embeddings.mamba_embedding = torch.compile(self.model.embeddings.mamba_embedding, mode="default", fullgraph=True)
            self.model.embeddings._aggregate_embeddings = torch.compile(self.model.embeddings._aggregate_embeddings, mode="reduce-overhead", dynamic=False, fullgraph=True)
        self.model.load_state_dict({k.replace('model.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}, strict=True)
        self.embedding_pulling = ckpt['hyper_parameters']['embedding_pulling']
    
    @batch
    @torch.inference_mode()
    def infer(self, inputs: np.ndarray, input_mask: np.ndarray):
        num_examples, num_channels, T, _ = inputs.shape
        inputs = inputs.reshape(num_examples * num_channels, T, 1)
        inputs = torch.from_numpy(process_ecg(inputs)).cuda()
        input_mask = torch.from_numpy(input_mask.astype(bool)).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embeddings_flat, extras = self.model(inputs, input_mask)
        embeddings_idx_flat = extras[0].b.values().reshape(num_examples, T).sum(dim=1)
        embeddings_idx = torch.cumsum(embeddings_idx_flat, dim=0)
        if self.embedding_pulling == 'last':
            embeddings = embeddings_flat.index_select(0, embeddings_idx - 1)
        elif self.embedding_pulling == 'mean':
            embeddings = torch.stack([t.mean(dim=0) for t in embeddings_flat.split(embeddings_idx_flat.tolist())])
        else:
            raise ValueError(f"Invalid embedding pulling method: {self.embedding_pulling}")
        return {"embeddings": embeddings.float().cpu().numpy()}


if __name__ == "__main__":
    with Triton(config=TritonConfig(http_port=8005, allow_grpc=False, allow_metrics=False)) as triton:
        model = EmbeddingModel()
        triton.bind(
            model_name="EmbeddingModel",
            infer_func=model.infer,
            inputs=[
                Tensor(name="inputs", shape=(-1, -1, 1), dtype=np.float32),
                Tensor(name="input_mask", shape=(-1,), dtype=np.int32),
            ],
            outputs=[Tensor(name="embeddings", shape=(-1, -1), dtype=np.float32)],
            config=ModelConfig(
                batching=True, 
                max_batch_size=32,
                batcher=DynamicBatcher(preferred_batch_size=[32], max_queue_delay_microseconds=1000000),
            ),
        )
        triton.serve()