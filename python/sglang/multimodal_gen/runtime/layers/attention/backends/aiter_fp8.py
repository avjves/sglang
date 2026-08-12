# SPDX-License-Identifier: Apache-2.0
#
# FP8 attention logic ported from xDiT:
# https://github.com/xdit-project/xDiT/blob/8672d0252760d8caa491bf4218038f83d635a6e0/xfuser/core/distributed/attention_backend.py
#
# Unlike xDiT (whose attention functions use a [batch, heads, seq, dim] layout
# and permute to [batch, seq, heads, dim] for the aiter kernel), SGL-D's
# ``AttentionImpl.forward`` already receives [batch, seq, heads, dim] -- the
# layout the aiter kernel expects -- so the permutes are dropped here. The
# Hadamard rotation only touches the last (head_dim) axis and is unaffected.

import inspect
import logging

import aiter
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

logger = logging.getLogger(__name__)

# Hadamard block size. Matches xDiT's hardcoded 128 and targets full-MHA
# head_dim==128 models (e.g. Wan self-/cross-attention).
_HADAMARD_BLOCK_R = 128


def _build_hadamard_matrix(
    block_r: int,
    dtype: torch.dtype = torch.bfloat16,
    allow_sylvester_fallback: bool = True,
) -> torch.Tensor | None:
    """Normalized Hadamard matrix (block_r x block_r, R @ R.T == I; block_r a
    power of two). Uses aiter's create_hadamard_matrix. If that's unavailable,
    falls back to a local Sylvester construction when allow_sylvester_fallback
    is set, otherwise returns None."""
    try:
        try:
            from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4 import (
                create_hadamard_matrix,
            )
        except ImportError:
            from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
                create_hadamard_matrix,
            )
        return create_hadamard_matrix(block_r, dtype=dtype) / (block_r**0.5)
    except ImportError:
        if not allow_sylvester_fallback:
            return None
        # Local Sylvester construction: H1=[[1]], H2n=[[Hn,Hn],[Hn,-Hn]].
        assert (
            block_r > 0 and (block_r & (block_r - 1)) == 0
        ), "Hadamard block_r must be a positive power of 2"
        H = torch.ones((1, 1), dtype=torch.float32)
        while H.shape[0] < block_r:
            H = torch.cat(
                [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
            )
        return (H / (block_r**0.5)).to(dtype)


def _replicate_hadamard_per_device(
    hadamard: torch.Tensor | None,
) -> dict[torch.device, torch.Tensor | None]:
    """Replicate a single Hadamard matrix on each available device, keyed by
    torch.device (all GPUs if CUDA is available, else CPU). A None matrix maps
    to None on every device."""
    if torch.cuda.is_available():
        devices = [
            torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
        ]
    else:
        devices = [torch.device("cpu")]
    return {
        device: (hadamard.to(device) if hadamard is not None else None)
        for device in devices
    }


def _aiter_hadamard_matrix(
    block_r: int, allow_sylvester_fallback: bool = True
) -> dict[torch.device, torch.Tensor | None]:
    """Build a normalized Hadamard matrix and replicate it across devices."""
    return _replicate_hadamard_per_device(
        _build_hadamard_matrix(
            block_r,
            dtype=torch.bfloat16,
            allow_sylvester_fallback=allow_sylvester_fallback,
        )
    )


FP8_HADAMARD_MATRIX = _aiter_hadamard_matrix(_HADAMARD_BLOCK_R)


def _fp8_hadamard_rotate(x: torch.Tensor, R: torch.Tensor | None) -> torch.Tensor:
    """Rotate the last (head_dim) axis by the Hadamard matrix R.

    Spreads outliers across dimensions to reduce quantization error while
    preserving attention scores (Q@K^T is invariant since R @ R.T == I).
    """
    if R is None:
        return x
    d = x.shape[-1]
    block_r = R.shape[-1]
    R = R.to(x.dtype)
    if block_r == d:
        return torch.matmul(x, R)
    return torch.matmul(x.unflatten(-1, (d // block_r, block_r)), R).flatten(-2)


def _aiter_fp8_has_descale() -> bool:
    """True if the installed aiter's flash_attn_fp8_pertensor_func accepts
    per-tensor descale vectors (q_descale/k_descale/v_descale)."""
    try:
        return (
            inspect.signature(aiter.flash_attn_fp8_pertensor_func).parameters.get(
                "q_descale"
            )
            is not None
        )
    except (AttributeError, TypeError):
        return False


AITER_FP8_HAS_DESCALE = _aiter_fp8_has_descale()


class AITerFP8Backend(AttentionBackend):
    """AITER FP8 attention backend (ROCm)."""

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER_FP8

    @staticmethod
    def get_impl_cls() -> type["AITerFP8Impl"]:
        return AITerFP8Impl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        # AITER FP8 backend does not require special metadata.
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError(
            "AITER FP8 backend does not have a metadata builder."
        )


class AITerFP8Impl(AttentionImpl):
    """FP8 attention via aiter.flash_attn_fp8_pertensor_func with Hadamard-rotated
    per-tensor quantization (ported from xDiT)."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        dropout_p: float = 0.0,
        **extra_impl_args,
    ) -> None:
        if num_kv_heads is not None and num_kv_heads != num_heads:
            raise NotImplementedError(
                "AITER FP8 backend does not support Grouped Query Attention yet."
            )
        self.causal = causal
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """
        Performs FP8 attention.

        Args:
            query: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            key: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            value: Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
            attn_metadata: Metadata for the attention operation (unused).

        Returns:
            Output tensor of shape [batch_size, seq_len, num_heads, head_dim]
        """
        R = FP8_HADAMARD_MATRIX[query.device]
        # Rotate Q and K only; V is quantized but not rotated. Q@K^T is preserved
        # because R @ R.T == I.
        query = _fp8_hadamard_rotate(query, R).contiguous()
        key = _fp8_hadamard_rotate(key, R).contiguous()
        value = value.contiguous()

        quant_dtype = aiter.dtypes.fp8
        dtype_max = torch.finfo(quant_dtype).max
        # xDiT default: dynamic per-tensor scale when descale vectors are
        # supported, otherwise a static scale of 1.0 (no descale).
        if AITER_FP8_HAS_DESCALE:
            scale = None
        else:
            scale = torch.tensor(1.0, dtype=torch.float32, device=query.device)

        quant_q, q_descale = aiter.per_tensor_quant(
            query, scale=scale, quant_dtype=quant_dtype, dtypeMax=dtype_max
        )
        quant_k, k_descale = aiter.per_tensor_quant(
            key, scale=scale, quant_dtype=quant_dtype, dtypeMax=dtype_max
        )
        quant_v, v_descale = aiter.per_tensor_quant(
            value, scale=scale, quant_dtype=quant_dtype, dtypeMax=dtype_max
        )

        descale_kwargs = {}
        if AITER_FP8_HAS_DESCALE:
            descale_kwargs = {
                "q_descale": q_descale,
                "k_descale": k_descale,
                "v_descale": v_descale,
            }

        return aiter.flash_attn_fp8_pertensor_func(
            quant_q,
            quant_k,
            quant_v,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
            **descale_kwargs,
        )

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "AITER FP8 backend does not support varlen attention."
        )
