from dataclasses import dataclass
from collections import defaultdict
from contextlib import contextmanager, nullcontext

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from .torchisms import torch, TT, nn, F, nested, NJT, summon_full_params
from .conceptual import BlockBoundaryMixin, get_seq_idx
from .config_hnet import HNetConfig
from .xf import Isotropic, Mamba2Simple
from .lin import Lin, HighPrecLinear, LMHead

### ################
### H-Net submodules
### ################


@dataclass(frozen=True)
class HNetExtra:
    b: TT  # (B,j1) boolean label for whether byte was selected
    loss_ratio: TT  # scalar tensor -- routing loss for this block
    compress_ratio: float  # scalar float -- compression ratio


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_func(x):
    return STE.apply(x)


class QProjPadded(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_flat: TT, w: TT, k_flat: TT, cu: TT):
        slen = x_flat.shape[0]
        # compute x@w.T, but padded left by 1seqlen
        q_padded = torch.empty(
            slen + 1, *x_flat.shape[1:], dtype=x_flat.dtype, device=x_flat.device
        )
        torch.mm(x_flat, w.T.type_as(x_flat), out=q_padded[1:])
        ctx.save_for_backward(x_flat, w, cu)
        return q_padded.index_copy_(0, cu[:-1], -k_flat[cu[:-1]])[:slen]

    @staticmethod
    def backward(ctx, dq_flat: TT):
        x_flat, w, cu = ctx.saved_tensors
        zero_grad = torch.zeros(
            cu.shape[0] - 1,
            dq_flat.shape[-1],
            device=dq_flat.device,
            dtype=dq_flat.dtype,
        )
        dq_flat = dq_flat.index_copy(0, cu[:-1], zero_grad)

        dx_flat = torch.zeros_like(x_flat)
        torch.mm(dq_flat[1:], w.type_as(dq_flat), out=dx_flat[:-1])
        dw = dq_flat[1:].mT @ x_flat[:-1]

        return dx_flat, dw, None, None


# NOTE: it's possible to fuse q/k/res proj into a single gemm kernel, but only iff they are of equal precision.
class RoutingModule(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.q_proj_layer = Lin(d, d)
        self.k_proj_layer = Lin(d, d)
        # https://github.com/goombalab/hnet/blob/main/hnet/modules/dc.py#L49
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d))
            self.k_proj_layer.weight.copy_(torch.eye(d))

    def forward(self, r_flat: TT, r_cu: TT):
        k_flat = self.k_proj_layer(r_flat)
        q_flat = QProjPadded.apply(r_flat, self.q_proj_layer.weight, k_flat, r_cu)
        cos_sim = F.cosine_similarity(q_flat, k_flat, dim=-1)
        p_flat = (0.5 - cos_sim / 2).clamp(0.0, 1.0)
        b_flat = p_flat >= 0.5
        p_select_cu = F.pad(b_flat.cumsum(0), (1, 0))[r_cu]
        return p_flat, b_flat, p_select_cu


class DeChunkLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        # for EMA scan kernel.
        self.block_size = 256
        self.headdim = 32
        self.nheads, _r = divmod(d, self.headdim)
        assert _r == 0
        A = -torch.ones(self.nheads, device="cuda", dtype=torch.float32)
        self.register_buffer("A", A, persistent=False)

    @staticmethod
    def forward_flat(
        h_flat: TT,
        b_flat: TT,
        p_selected_flat: TT,
        h_seq_idx: TT,
        *,
        eps=1e-4,
        nheads: int,
        headdim: int,
        block_size: int,
        A: TT,
    ):
        p = p_selected_flat.float().clamp(eps, 1 - eps)

        dt = -torch.log1p(-p.float())[..., None]
        h = (h_flat.float() / dt).type_as(h_flat)
        c = torch.ones_like(p := p.type_as(h)[None, :, None, None])

        z_bar_flat = mamba_chunk_scan_combined(
            h.view(1, -1, nheads, headdim),
            dt.expand(-1, nheads).to(h.dtype)[None],
            A,
            p,
            c,
            chunk_size=block_size,
            seq_idx=h_seq_idx,
        )[0].view(-1, h.shape[-1])

        inner2outer_idx = b_flat.cumsum(0) - 1
        return z_bar_flat.index_select(0, inner2outer_idx)

    def forward(
        self, h_flat: TT, b_flat: TT, p_selected_flat: TT, h_seq_idx: TT, *, eps=1e-4
    ):
        return self.forward_flat(
            h_flat,
            b_flat,
            p_selected_flat,
            h_seq_idx,
            eps=eps,
            nheads=self.nheads,
            headdim=self.headdim,
            block_size=self.block_size,
            A=self.A,
        )


### #################
### Final HNet Module
### #################


class HNet(nn.Module):
    def __init__(self, c: HNetConfig, stage_idx: int):
        super().__init__()
        self.stage_idx = stage_idx
        self.d = c.d_model[stage_idx]
        try:
            self.n = c.N_compress[stage_idx + 1] / c.N_compress[stage_idx]
        except IndexError:
            self.n = None

        arch_layout = c.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]

        assert len(arch_layout) in [3, 1]
        self.is_innermost = len(arch_layout) == 1

        if self.is_innermost:
            self.main_network = Isotropic(
                c, arch_layout[0], stage_idx=stage_idx
            )  # <-- don't increment
        else:
            self.encoder = Isotropic(c, arch_layout[0], stage_idx=stage_idx)
            self.main_network = HNet(c, stage_idx + 1)
            self.decoder = Isotropic(c, arch_layout[2], stage_idx=stage_idx)

            self.routing_module = RoutingModule(self.d)
            self.dechunk_layer = DeChunkLayer(self.d)
            self.residual_proj = HighPrecLinear(self.d, self.d)

        d_gain = self.d - c.d_model[stage_idx - 1] if stage_idx else None
        self.pad_dimension = (
            nn.Parameter(torch.zeros(d_gain, device="cuda")) if d_gain else None
        )

    # only compile blocks within a hnet, not the hnet itself
    def block_compile(self, ac: bool):
        self.main_network.block_compile(ac)
        if self.is_innermost:
            return
        self.encoder.block_compile(ac)
        self.decoder.block_compile(ac)
        self.register_module(
            "routing_module",
            torch.compile(self.routing_module, backend="inductor", fullgraph=True),
        )
        self.register_module(
            "residual_proj",
            torch.compile(self.residual_proj, backend="inductor", fullgraph=True),
        )
        self.ratio_loss = torch.compile(
            self.ratio_loss, backend="inductor", fullgraph=True, dynamic=True
        )

    def ratio_loss(self, b_flat: TT, p_flat: TT):
        assert self.n, "HNetConfig did not receive valid N_compress; please edit it"
        l = b_flat.numel()
        f = b_flat.sum().float() / l
        g = p_flat.float().sum() / l
        drop_experts = self.n * (1 - f) * (1 - g) / (self.n - 1)
        keep_expert = self.n * f * g
        return keep_expert + drop_experts

    @contextmanager
    def least_blocking_masked_select(
        self, *outer_flat_tensors: list[TT], mask_flat: TT, cu_seqlens: TT
    ):
        # WARNING: do not try to compile this. inductor will just wipe all pin memory & copy & Event & etc.
        inner_stats_cuda = torch.stack([cu_seqlens.diff().max(), cu_seqlens[-1]])
        inner_stats_cpu = torch.empty_like(
            inner_stats_cuda, device="cpu", pin_memory=True
        )
        inner_stats_cpu.copy_(inner_stats_cuda, non_blocking=True)
        d2h_event = torch.cuda.Event()
        d2h_event.record()

        # in the yield region, the end-user is expected to enqueue as much GPU work as possible, to make the CPU sync cheap.
        yield (mutable_res := []), inner_stats_cpu

        d2h_event.synchronize()
        inner_flatlen = inner_stats_cpu[1].item()
        idx_flat = mask_flat.nonzero_static(size=inner_flatlen).squeeze(-1)
        for outer in outer_flat_tensors:
            mutable_res.append(outer.index_select(0, idx_flat))

    def forward(self, x_flat: TT, flat_cu: TT, msl: int):
        d_orig = x_flat.shape[-1]
        x_flat = (
            x_flat
            if self.pad_dimension is None
            else torch.cat(
                [x_flat, self.pad_dimension.expand(x_flat.shape[0], -1)], dim=-1
            )
        )
        x_flat = x_flat.bfloat16()

        if self.is_innermost:
            return self.main_network(x_flat, flat_cu, msl)[..., :d_orig], []

        r_flat = self.encoder(x_flat, flat_cu, msl)
        p_flat, b_flat, select_cu = self.routing_module(r_flat, flat_cu)

        # print('select_cu', select_cu)
        # obtaining r_select/p_select would require a cpu-sync'ing .masked_select in normal circumstances.
        # To avoid this, we initiate a D2H of the inner H-Net's seqlen ASAP, and enqueue work to let the GPU race ahead.
        # Note that, if you are **already CPU bound** prior to this (e.g. in really small runs), this code is detrimental.
        with self.least_blocking_masked_select(
            p_flat, r_flat, mask_flat=b_flat, cu_seqlens=select_cu
        ) as (pending_selected_tensors, pending_cpu_stats):
            ratio_loss = (
                self.ratio_loss(b_flat, p_flat) if torch.is_grad_enabled() else 0
            )
            c_flat = torch.where(b_flat, p_flat, 1 - p_flat)[..., None]
            residual = self.residual_proj(r_flat)
        p_select, r_select = pending_selected_tensors

        h_select, extras = self.main_network(
            r_select, select_cu, pending_cpu_stats[0].item()
        )

        x_flat = self.dechunk_layer(
            h_select, b_flat, p_select, get_seq_idx(select_cu, p_select.shape[0])
        )
        x_flat = (residual + x_flat.float() * ste_func(c_flat)).type_as(x_flat)
        x_flat = self.decoder(x_flat, flat_cu, msl)[..., :d_orig]

        extra = HNetExtra(
            nested.nested_tensor_from_jagged(b_flat, flat_cu, max_seqlen=msl),
            ratio_loss,
            p_select.numel() / p_flat.numel(),
        )

        return x_flat, [extra] + extras


class HNetEncoder(nn.Module):
    """
    HNetEncoder is an encoder-only version of HNet that outputs raw main_network outputs
    without applying dechunking, residual addition, or decoder modules. This is useful for 
    tasks that need the raw encoded representations at the compressed sequence length.
    
    Note: The outputs are at the compressed/selected sequence length (select_cu), not the
    original input sequence length, since we skip the dechunking step.
    
    When use_decoder=True, it also returns decoder outputs for masked prediction tasks.
    """
    def __init__(self, c: HNetConfig, stage_idx: int, finetune_mode: bool = False):
        super().__init__()
        self.stage_idx = stage_idx
        self.d = c.d_model[stage_idx]
        self.use_decoder = c.use_decoder
        self.finetune_mode = finetune_mode
        try:
            self.n = c.N_compress[stage_idx + 1] / c.N_compress[stage_idx]
        except IndexError:
            self.n = None

        arch_layout = c.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]

        assert len(arch_layout) in [3, 1]
        self.is_innermost = len(arch_layout) == 1

        if self.is_innermost:
            self.main_network = Isotropic(
                c, arch_layout[0], stage_idx=stage_idx
            )  # <-- don't increment
        else:
            self.encoder = Isotropic(c, arch_layout[0], stage_idx=stage_idx)
            self.main_network = HNetEncoder(c, stage_idx + 1)  # Use HNetEncoder recursively
            
            self.routing_module = RoutingModule(self.d)
            
            # Conditionally initialize decoder components for masked prediction
            if self.use_decoder:
                self.decoder = Isotropic(c, arch_layout[2], stage_idx=stage_idx)
                self.dechunk_layer = DeChunkLayer(self.d)
                self.residual_proj = HighPrecLinear(self.d, self.d)

        d_gain = self.d - c.d_model[stage_idx - 1] if stage_idx else None
        self.pad_dimension = (
            nn.Parameter(torch.zeros(d_gain, device="cuda")) if d_gain else None
        )

    # only compile blocks within a hnet, not the hnet itself
    def block_compile(self, ac: bool):
        self.main_network.block_compile(ac)
        if self.is_innermost:
            return
        self.encoder.block_compile(ac)
        if self.use_decoder:
            self.decoder.block_compile(ac)
            self.register_module(
                "residual_proj",
                torch.compile(self.residual_proj, backend="inductor", fullgraph=True),
            )
        self.register_module(
            "routing_module",
            torch.compile(self.routing_module, backend="inductor", fullgraph=True),
        )
        self.ratio_loss = torch.compile(
            self.ratio_loss, backend="inductor", fullgraph=True, dynamic=True
        )

    def ratio_loss(self, b_flat: TT, p_flat: TT):
        assert self.n, "HNetConfig did not receive valid N_compress; please edit it"
        l = b_flat.numel()
        f = b_flat.sum().float() / l
        g = p_flat.float().sum() / l
        drop_experts = self.n * (1 - f) * (1 - g) / (self.n - 1)
        keep_expert = self.n * f * g
        return keep_expert + drop_experts

    @contextmanager
    def least_blocking_masked_select(
        self, *outer_flat_tensors: list[TT], mask_flat: TT, cu_seqlens: TT
    ):
        # WARNING: do not try to compile this. inductor will just wipe all pin memory & copy & Event & etc.
        inner_stats_cuda = torch.stack([cu_seqlens.diff().max(), cu_seqlens[-1]])
        inner_stats_cpu = torch.empty_like(
            inner_stats_cuda, device="cpu", pin_memory=True
        )
        inner_stats_cpu.copy_(inner_stats_cuda, non_blocking=True)
        d2h_event = torch.cuda.Event()
        d2h_event.record()

        # in the yield region, the end-user is expected to enqueue as much GPU work as possible, to make the CPU sync cheap.
        yield (mutable_res := []), inner_stats_cpu

        d2h_event.synchronize()
        inner_flatlen = inner_stats_cpu[1].item()
        idx_flat = mask_flat.nonzero_static(size=inner_flatlen).squeeze(-1)
        for outer in outer_flat_tensors:
            mutable_res.append(outer.index_select(0, idx_flat))

    def forward(self, x_flat: TT, flat_cu: TT, msl: int):
        d_orig = x_flat.shape[-1]
        x_flat = (
            x_flat
            if self.pad_dimension is None
            else torch.cat(
                [x_flat, self.pad_dimension.expand(x_flat.shape[0], -1)], dim=-1
            )
        )
        x_flat = x_flat.bfloat16()

        if self.is_innermost:
            # For innermost, just return main_network output
            # Note: innermost stage has no decoder architecture (arch_layout length is 1)
            h_select = self.main_network(x_flat, flat_cu, msl)[..., :d_orig]
            return h_select, []
        
        with torch.inference_mode() if self.finetune_mode else nullcontext():
            r_flat = self.encoder(x_flat, flat_cu, msl)
        p_flat, b_flat, select_cu = self.routing_module(r_flat, flat_cu)

        # obtaining r_select/p_select would require a cpu-sync'ing .masked_select in normal circumstances.
        # To avoid this, we initiate a D2H of the inner H-Net's seqlen ASAP, and enqueue work to let the GPU race ahead.
        # Note that, if you are **already CPU bound** prior to this (e.g. in really small runs), this code is detrimental.
        with self.least_blocking_masked_select(
            p_flat, r_flat, mask_flat=b_flat, cu_seqlens=select_cu
        ) as (pending_selected_tensors, pending_cpu_stats):
            ratio_loss = (
                self.ratio_loss(b_flat, p_flat) if torch.is_grad_enabled() else 0
            )
            if self.use_decoder:
                c_flat = torch.where(b_flat, p_flat, 1 - p_flat)[..., None]
                residual = self.residual_proj(r_flat)
        p_select, r_select = pending_selected_tensors

        
        main_network_output = self.main_network(
            r_select, select_cu, pending_cpu_stats[0].item()
        )
        
        # Handle recursive HNetEncoder output: may return tuple if decoder is enabled at deeper stage
        # We only need h_select for processing at this stage, decoder outputs from deeper stages are not used here
        if isinstance(main_network_output[0], tuple):
            h_select, _ = main_network_output[0]  # Extract h_select from tuple
            extras = main_network_output[1]
        else:
            h_select, extras = main_network_output

        # Return raw main_network outputs for similarity pretraining
        # Note: h_select is at compressed sequence length (select_cu), not original length
        # h_select dimension matches the next stage's d_model, not d_orig

        extra = HNetExtra(
            nested.nested_tensor_from_jagged(b_flat, flat_cu, max_seqlen=msl),
            ratio_loss,
            p_select.numel() / p_flat.numel(),
        )

        # If decoder is enabled, also compute decoder outputs for masked prediction
        if self.use_decoder:
            x_flat_dec = self.dechunk_layer(
                h_select, b_flat, p_select, get_seq_idx(select_cu, p_select.shape[0])
            )
            x_flat_dec = (residual + x_flat_dec.float() * ste_func(c_flat)).type_as(x_flat_dec)
            decoder_output = self.decoder(x_flat_dec, flat_cu, msl)[..., :d_orig]
            return (h_select, decoder_output), [extra] + extras

        return h_select, [extra] + extras


class Mamba1DEmbedding(nn.Module):
    def __init__(self, inner_dim: int, out_dim: int):
        super().__init__()
        self.embedding = nn.Linear(1, inner_dim)
        self.mamba_embedding = Mamba2Simple(inner_dim, d_state=16, d_conv=4, expand=2, headdim=8)
        self.lift = nn.Linear(inner_dim, out_dim, bias=False)

    def forward(self, inputs: TT, input_mask: TT):
        assert input_mask.shape[1] == 1, "num_channels_per_example must be all 1 for mamba1d embedding"
        b, t, _ = inputs.shape
        inputs = self.embedding(inputs)
        cu_s = torch.arange(t, (b + 1) * t, t, device=inputs.device)
        x_flat = inputs.view(-1, inputs.shape[2])
         # Pad x_flat to a multiple of 128 for kernel efficiency (same as Isotropic)
        REQUIRED_PAD_MODULO = 128
        original_cu_s = cu_s  # Store original for unpadding
        if padding := (-x_flat.shape[0]) % REQUIRED_PAD_MODULO:
            cu_s = F.pad(cu_s, (0, 1), value=padding + x_flat.shape[0])
            x_flat = F.pad(x_flat, (0, 0, 0, padding), value=0)
        
        # Convert cu_seqlens to seq_idx for Mamba2Simple (similar to how encoder uses it)
        seq_idx = get_seq_idx(cu_s, x_flat.shape[0])
        x_flat = self.mamba_embedding(x_flat[None], seq_idx=seq_idx)[0]
        x_flat = self.lift(x_flat)
        
        # Unpad before passing to backbone (backbone will apply its own padding)
        if padding:
            x_flat = x_flat[:x_flat.shape[0] - padding]
            cu_s = original_cu_s  # Restore original cu_s 
        return x_flat.view(b, t, -1)


class MambaAttentionMultichannelEmbedding(nn.Module):
    def __init__(self, inner_dim: int, out_dim: int):
        super().__init__()
        self.inner_dim = inner_dim
        self.out_dim = out_dim
        self.embeddings = nn.Linear(1, inner_dim)
        self.mamba_embedding = Mamba2Simple(inner_dim, d_state=16, d_conv=4, expand=2, headdim=8)
        # Project mamba output to consistent dimension for cross-attention
        self.lift = nn.Linear(inner_dim, out_dim, bias=False)
        # Cross-attention to fuse channels for each example
        # num_heads: use 8 heads by default, adjust if needed
        num_heads = 8
        assert out_dim % num_heads == 0, f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.output_proj = nn.Linear(out_dim, out_dim, bias=False)
        # Normalization layers similar to attention blocks
        self.pre_attn_norm = nn.RMSNorm(out_dim, eps=1e-5)
        self.post_proj_norm = nn.RMSNorm(out_dim, eps=1e-5)
        # Learnable query embeddings for cross-attention (one per example)
        # These will be used to aggregate channel information
        self.query_embeddings = nn.Parameter(torch.randn(1, 1, out_dim))


    def forward(self, inputs: TT, input_mask: TT):
        # Mask out padded channels from dataset to prevent gradient flow
        # Reshape to identify which channels are padded per example
        num_examples, max_channels = input_mask.shape

        b, t, _ = inputs.shape
        inputs = self.embeddings(inputs)
        inputs = inputs * (~input_mask.view(-1)).unsqueeze(-1).unsqueeze(-1)
        
        cu_s = torch.arange(0, (b + 1) * t, t, device=inputs.device)
        x_flat = inputs.view(-1, inputs.shape[2])
         # Pad x_flat to a multiple of 128 for kernel efficiency (same as Isotropic)
        REQUIRED_PAD_MODULO = 128
        original_cu_s = cu_s  # Store original for unpadding
        if padding := (-x_flat.shape[0]) % REQUIRED_PAD_MODULO:
            cu_s = F.pad(cu_s, (0, 1), value=padding + x_flat.shape[0])
            x_flat = F.pad(x_flat, (0, 0, 0, padding), value=0)
        
        # Convert cu_seqlens to seq_idx for Mamba2Simple (similar to how encoder uses it)
        seq_idx = get_seq_idx(cu_s, x_flat.shape[0])
        x_flat = self.mamba_embedding(x_flat[None], seq_idx=seq_idx)[0]
        x_flat = self.lift(x_flat)
        
        # Unpad before passing to cross-attention
        if padding:
            x_flat = x_flat[:x_flat.shape[0] - padding]
            cu_s = original_cu_s  # Restore original cu_s
        
        # Build batched tensor: (num_examples, max_channels, t, out_dim)
        batched_embeddings = x_flat.view(num_examples, -1, t, self.out_dim)
        batched_embeddings = batched_embeddings * (~input_mask).unsqueeze(-1).unsqueeze(-1)
        
        # Permute to (num_examples, t, max_channels, out_dim) for attention
        batched_embeddings = batched_embeddings.permute(0, 2, 1, 3)  # (num_examples, t, max_channels, out_dim)
        # Reshape to (num_examples * t, max_channels, out_dim) for batched attention
        batched_embeddings = batched_embeddings.reshape(num_examples * t, max_channels, self.out_dim)
        
        # Apply pre-attention normalization to key/value embeddings
        batched_embeddings = self.pre_attn_norm(batched_embeddings)
        
        # Expand query: (num_examples * t, 1, out_dim)
        query = self.query_embeddings.expand(num_examples * t, 1, -1)
        
        # Create key padding mask: (num_examples * t, max_channels)
        # Repeat mask for each timestep (same mask for all timesteps in an example)
        key_padding_mask_expanded = input_mask.unsqueeze(1).expand(num_examples, t, max_channels)
        key_padding_mask_expanded = key_padding_mask_expanded.reshape(num_examples * t, max_channels)
        
        # Apply batched cross-attention
        attn_output, _ = self.cross_attn(query, batched_embeddings, batched_embeddings, 
                                        key_padding_mask=key_padding_mask_expanded)  # (num_examples * t, 1, out_dim)
        combined_embeddings = attn_output.squeeze(1)  # (num_examples * t, out_dim)
        combined_embeddings = self.output_proj(combined_embeddings)
        # Apply post-projection normalization
        combined_embeddings = self.post_proj_norm(combined_embeddings)
        return combined_embeddings.reshape(num_examples, t, -1)


class GatedMambaMultichannelEmbedding(nn.Module):
    def __init__(self, inner_dim: int, out_dim: int):
        super().__init__()
        self.inner_dim = inner_dim
        self.out_dim = out_dim
        self.embeddings = nn.Linear(1, inner_dim)
        self.mamba_embedding = Mamba2Simple(inner_dim, d_state=16, d_conv=4, expand=2, headdim=8)
        self.lift = nn.Linear(inner_dim, out_dim, bias=False)
        self.gate = nn.Linear(inner_dim, 1, bias=False)

        self.output_proj = nn.Linear(out_dim, out_dim, bias=False)
        # Normalization layers similar to attention blocks
        self.pre_norm = nn.RMSNorm(out_dim, eps=1e-5)
        self.post_proj_norm = nn.RMSNorm(out_dim, eps=1e-5)
    
    def _aggregate_embeddings(self, batched_inner: TT, input_mask: TT):
        # Compute gate scores: (num_examples, max_channels, t, 1)
        batched_scores = self.gate(batched_inner)  # (num_examples, max_channels, t, 1)
        
        # Lift embeddings to output dimension
        batched_embeddings = self.lift(batched_inner)  # (num_examples, max_channels, t, out_dim)
        batched_embeddings = batched_embeddings * (~input_mask).unsqueeze(-1).unsqueeze(-1)
        
        # Apply mask to scores (set padding channels to -inf before softmax)
        batched_scores = batched_scores.squeeze(-1)  # (num_examples, max_channels, t)
        
        # Use torch.where instead of masked_fill to avoid CUDA graph recapture
        # torch.where with scalar is more CUDA graph friendly than masked_fill with float('-inf')
        mask_expanded = input_mask.unsqueeze(-1)  # (num_examples, max_channels, 1)
        batched_scores = torch.where(mask_expanded, -torch.inf, batched_scores)
        
        # Permute to (num_examples, t, max_channels, 1) and apply softmax
        batched_scores = batched_scores.permute(0, 2, 1).unsqueeze(-1)  # (num_examples, t, max_channels, 1)
        batched_scores = torch.softmax(batched_scores, dim=2)  # (num_examples, t, max_channels, 1)
        
        # Permute embeddings to (num_examples, t, max_channels, out_dim) for weighted aggregation
        batched_embeddings = batched_embeddings.permute(0, 2, 1, 3)  # (num_examples, t, max_channels, out_dim)
        
        # Apply pre-aggregation normalization
        batched_embeddings = self.pre_norm(batched_embeddings)
        
        # Apply softmax weights: (num_examples, t, max_channels, out_dim) * (num_examples, t, max_channels, 1)
        # Sum over channel dimension to get aggregated embeddings
        combined_embeddings = (batched_embeddings * batched_scores).sum(dim=2)  # (num_examples, t, out_dim)
        
        # Apply output projection and post-projection normalization
        combined_embeddings = self.output_proj(combined_embeddings)
        combined_embeddings = self.post_proj_norm(combined_embeddings)
        
        return combined_embeddings

    def forward(self, inputs: TT, input_mask: TT):
        num_examples, max_channels = input_mask.shape

        b, t, _ = inputs.shape
        inputs = self.embeddings(inputs)
        inputs = inputs * (~input_mask.view(-1)).unsqueeze(-1).unsqueeze(-1)

        cu_s = torch.arange(0, (b + 1) * t, t, device=inputs.device)
        x_flat = inputs.view(-1, inputs.shape[2])
            # Pad x_flat to a multiple of 128 for kernel efficiency (same as Isotropic)
        REQUIRED_PAD_MODULO = 128
        original_cu_s = cu_s  # Store original for unpadding
        if padding := (-x_flat.shape[0]) % REQUIRED_PAD_MODULO:
            cu_s = F.pad(cu_s, (0, 1), value=padding + x_flat.shape[0])
            x_flat = F.pad(x_flat, (0, 0, 0, padding), value=0)

        # Convert cu_seqlens to seq_idx for Mamba2Simple (similar to how encoder uses it)
        seq_idx = get_seq_idx(cu_s, x_flat.shape[0])
        x_flat = self.mamba_embedding(x_flat[None], seq_idx=seq_idx)[0]

        # Unpad before passing to cross-attention
        if padding:
            x_flat = x_flat[:x_flat.shape[0] - padding]
            cu_s = original_cu_s  # Restore original cu_s
        
        # Build batched tensor for inner_dim: (num_examples, max_channels, t, inner_dim)
        batched_inner = x_flat.view(num_examples, -1, t, self.inner_dim)
        return self._aggregate_embeddings(batched_inner, input_mask)
        



class HNetTS(BlockBoundaryMixin, nn.Module):
    def __init__(self, c: HNetConfig, finetune_mode: bool = False):
        super().__init__()
        self.c = c
        self.use_decoder = c.use_decoder
        d = c.d_model[0]
        self.embedding_type = c.embedding_type
        embedding_config = c.embedding_config
        if self.embedding_type == 'simple':
            self.embeddings = nn.Linear(1, d)
        elif self.embedding_type == 'mamba1d':
            inner_dim = embedding_config.get('inner_dim', 32)
            self.embeddings = Mamba1DEmbedding(inner_dim, d)
        elif self.embedding_type == 'mamba_attention_multichannel':
            inner_dim = embedding_config.get('inner_dim', 32)
            self.embeddings = MambaAttentionMultichannelEmbedding(inner_dim, d)
        elif self.embedding_type == 'gated_mamba_multichannel':
            inner_dim = embedding_config.get('inner_dim', 32)
            self.embeddings = GatedMambaMultichannelEmbedding(inner_dim, d)
        else:
            raise ValueError(f"Invalid embedding type: {self.embedding_type}")
        self.backbone = HNetEncoder(c, stage_idx=0, finetune_mode=finetune_mode)
        if self.c.use_decoder:
            self.ts_head = nn.Linear(d, 1)
        self.finetune_mode = finetune_mode


    def forward(self, inputs: TT, input_mask: TT):
        assert inputs.ndim == 3 and inputs.shape[2] == 1, "inputs must be a 3D tensor with shape (batch_size, sequence_length, 1)"

        with torch.inference_mode() if self.finetune_mode else nullcontext:
            if self.embedding_type == 'simple':
                assert torch.all(input_mask.shape[1] == 1), "channel_mask must be all 1 for simple embedding"
                inputs = self.embeddings(inputs)
            else:
                inputs = self.embeddings(inputs, input_mask)
            
            inputs = nested.nested_tensor(inputs, layout=torch.jagged)
            cu_s, msl = inputs.offsets(), inputs._max_seqlen
            x_flat = inputs.values()

        backbone_output = self.backbone(x_flat, cu_s, msl)
        # Handle both cases: with decoder (returns tuple) and without decoder (returns single tensor)
        if self.use_decoder:
            h_select, decoder_output = backbone_output[0]
            decoder_output = self.ts_head(decoder_output)
            extra = backbone_output[1]
            return (h_select, decoder_output), extra
        else:
            h_select, extra = backbone_output
            return h_select, extra
    
    def split_params_by_hierachy(self) -> list[list[nn.Parameter]]:
        # for each param, count the number of times ".main_network" appears in it.
        d = defaultdict(list)
        for n, p in self.named_parameters():
            d[n.count("main_network")].append(p)
        # special-case innermost hnet which has redundant .main_network
        max_depth = max(d.keys())
        assert 1 == len(d[max_depth - 1]), (
            f"expected single .pad_dimension at {max_depth - 1}"
        )
        d[max_depth - 1] += d.pop(max_depth)

        return [d[k] for k in range(len(d))]


class HNetLM(BlockBoundaryMixin, nn.Module):
    def __init__(self, c: HNetConfig):
        super().__init__()
        self.c, v, d = c, c.vocab_size, c.d_model[0]
        self.embeddings = nn.Embedding(v, d)
        self.backbone = HNet(c, stage_idx=0)
        self.lm_head = LMHead(d, v)

    # Top-level contract:
    # 1. if lbls is provided, return (loss_mean,loss_sum),extras[]
    #    use loss_mean for autograd, and loss_sum for bpb calc.
    #    use extras[] to grab ratio loss && log compression ratio
    # 2. if lbls is None,     return logits,extras[]
    #    use logits for autoregressive sampling.
    #    use extras[] to grab selected token IDs (b) for sampling pretty-printing.
    def forward(
        self, iids: TT, lbls: TT | None = None
    ) -> tuple[TT | tuple[TT, TT], list]:
        assert iids.is_nested and iids.ndim == 2
        cu_s, msl = iids.offsets(), iids._max_seqlen
        x_flat = self.embeddings(iids.values())
        x_flat, extra = self.backbone(x_flat, cu_s, msl)
        res = self.lm_head(x_flat, lbls if lbls is None else lbls.values())
        if lbls is None:
            res = nested.nested_tensor_from_jagged(res, cu_s, max_seqlen=msl)
        return res, extra

    def split_params_by_hierachy(self) -> list[list[nn.Parameter]]:
        # for each param, count the number of times ".main_network" appears in it.
        d = defaultdict(list)
        for n, p in self.named_parameters():
            d[n.count("main_network")].append(p)
        # special-case innermost hnet which has redundant .main_network
        max_depth = max(d.keys())
        assert 1 == len(d[max_depth - 1]), (
            f"expected single .pad_dimension at {max_depth - 1}"
        )
        d[max_depth - 1] += d.pop(max_depth)

        return [d[k] for k in range(len(d))]

    def load_goomba_ckpt(self, path: str | None):
        from omegaconf import ListConfig

        if path is None:
            return
        with torch.serialization.safe_globals([ListConfig]):
            d = torch.load(path, mmap=True, weights_only=False)
        self.load_state_dict(d)

    @contextmanager
    def sampling_mode(self):
        with (
            summon_full_params(self),
            torch.compiler.set_stance("force_eager"),
            torch.autocast("cuda", torch.bfloat16, cache_enabled=False),
        ):
            yield


def test_fwd_correctness():
    import re
    from .sampling import ByteTokenizer, completion_sync

    ## load hardcoded model
    c = HNetConfig.load_config("hnet_2stage_XL.json")
    t = ByteTokenizer()
    with torch.device("cuda"):
        m = HNetLM(c).bfloat16()
    m.load_goomba_ckpt("hnet_2stage_XL.pt")

    ## check randint fwd logits
    torch.manual_seed(0)
    iids = torch.randint(0, 256, (77,), dtype=torch.long, device="cuda")
    with torch.no_grad():
        pfill = m(NJT([iids]))[0].values()
    # original: "tensor[77, 256] bf16 n=19712 (38Kb) x∈[-12.562, 15.188] μ=-2.047 σ=3.875 cuda:0"
    assert -12.75 < pfill.min().item() < -12.25 and 15 < pfill.max().item() < 15.5, (
        pfill
    )
    assert -2.1 < pfill.mean().item() < -2.0 and 3.87 < pfill.std().item() < 3.88, pfill
    print(f"{pfill=}")

    ## check greedy sampling result
    comp = completion_sync("Hello world!", t, m, max_new=200, temp=0.0001, min_p=0.0001)
    comp = re.sub(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]", "", comp)  # )
    assert (
        comp
        == " I hope you are doing well. In this article, we will discuss the basics of the Python programming language. We will start with the basics of Python and then move on to more advanced topics. So, let"
    ), comp


__all__ = ["HNetLM", "HNetEncoder"]
if __name__ == "__main__":
    test_fwd_correctness()
