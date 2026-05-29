# SPDX-License-Identifier: Apache-2.0

import inspect
import logging

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

logger = logging.getLogger(__name__)


def _check_fp8_has_descale() -> bool:
    """Check whether the installed aiter exposes descale params."""
    try:
        import aiter

        return (
            inspect.signature(aiter.flash_attn_fp8_pertensor_func)
            .parameters.get("q_descale")
            is not None
        )
    except (AttributeError, TypeError, ImportError):
        return False


_FP8_HAS_DESCALE: bool = _check_fp8_has_descale()


class AITERFP8Backend(AttentionBackend):
    """Backend for AITER FP8 per-tensor quantized flash attention."""

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER_FP8

    @staticmethod
    def get_impl_cls() -> type["AITERFP8Impl"]:
        return AITERFP8Impl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError(
            "AITER FP8 backend does not have a metadata builder."
        )


class AITERFP8Impl(AttentionImpl):
    """
    FP8 per-tensor quantized attention via ``aiter.flash_attn_fp8_pertensor_func``.

    Q/K/V are dynamically quantized to FP8 (e4m3fn) before the kernel call.
    If the installed aiter version supports descale vectors they are forwarded;
    otherwise a static scale of 1.0 is used (matching xDiT behaviour).
    """

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
        self.causal = causal
        self.softmax_scale = softmax_scale

    @torch.compiler.disable
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """
        Quantize Q/K/V to FP8 and run ``aiter.flash_attn_fp8_pertensor_func``.

        Args:
            query:  [B, S, H, D]
            key:    [B, S, H, D]
            value:  [B, S, H, D]
            attn_metadata: unused.

        Returns:
            Output tensor [B, S, H, D].
        """
        import aiter

        quant_dtype = aiter.dtypes.fp8
        dtype_max = torch.finfo(quant_dtype).max

        # Dynamic scaling (scale=None) when descale vectors are available;
        # static scale=1.0 otherwise – mirrors xDiT's default behaviour.
        scale = None if _FP8_HAS_DESCALE else torch.tensor(
            1.0, dtype=torch.float32, device=query.device
        )

        quant_q, q_descale = aiter.per_tensor_quant(
            query, scale=scale, quant_dtype=quant_dtype, dtypeMax=dtype_max
        )
        quant_k, k_descale = aiter.per_tensor_quant(
            key, scale=scale, quant_dtype=quant_dtype, dtypeMax=dtype_max
        )
        quant_v, v_descale = aiter.per_tensor_quant(
            value, scale=scale, quant_dtype=quant_dtype, dtypeMax=dtype_max
        )

        kwargs = {}
        if _FP8_HAS_DESCALE:
            kwargs = {
                "q_descale": q_descale,
                "k_descale": k_descale,
                "v_descale": v_descale,
            }

        output = aiter.flash_attn_fp8_pertensor_func(
            quant_q,
            quant_k,
            quant_v,
            causal=self.causal,
            **kwargs,
        )

        return output
