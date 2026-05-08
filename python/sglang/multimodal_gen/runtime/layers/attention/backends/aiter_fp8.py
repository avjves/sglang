# SPDX-License-Identifier: Apache-2.0
# Adapted from xDiT AITER_FP8 attention backend.

import inspect
import os

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class AITerFP8Backend(AttentionBackend):
    """
    Backend for AITER FP8 per-tensor quantized attention.
    """

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER_FP8

    @staticmethod
    def get_impl_cls() -> type["AITerFP8Impl"]:
        return AITerFP8Impl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError(
            "AITer FP8 backend does not have a metadata builder."
        )


class AITerFP8Impl(AttentionImpl):
    """
    FP8 attention using AITER's per-tensor quantized flash attention kernel.

    Quantizes Q, K, V to FP8 on the fly, then calls
    ``aiter.flash_attn_fp8_pertensor_func``.  Supports both dynamic scaling
    (default) and static scaling via the ``SGLANG_AITER_FP8_STATIC_SCALE``
    environment variable.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        if num_kv_heads is not None and num_kv_heads != num_heads:
            raise NotImplementedError(
                "AITer FP8 backend does not support Grouped Query Attention yet."
            )

        try:
            import aiter

            self._per_tensor_quant = aiter.per_tensor_quant
            self._fp8_attn_fn = aiter.flash_attn_fp8_pertensor_func
            self._fp8_dtype = aiter.dtypes.fp8
        except ImportError:
            raise ImportError(
                "AITER FP8 attention is not available. "
                "Please install or update the AITER package."
            )

        self.causal = causal
        self._dtype_max = torch.finfo(self._fp8_dtype).max

        # Check whether the installed AITER version supports descale vectors.
        self._has_descale = (
            inspect.signature(self._fp8_attn_fn)
            .parameters.get("q_descale")
            is not None
        )

        # Parse optional static scale from environment.
        self._static_scale = self._parse_static_scale()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_static_scale() -> float | None:
        raw = os.environ.get("SGLANG_AITER_FP8_STATIC_SCALE")
        if raw is None:
            return None
        try:
            val = float(raw)
            return val if val > 1.0 else None
        except (TypeError, ValueError):
            return None

    def _get_scale_tensor(self, device: torch.device) -> torch.Tensor | None:
        if self._has_descale:
            if self._static_scale is None:
                return None  # dynamic scaling
            return torch.tensor(
                self._static_scale, dtype=torch.float32, device=device
            )
        # Older AITER without descale — must use scale=1.0.
        return torch.tensor(1.0, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """
        Performs FP8-quantized attention.

        Args:
            query: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            key: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            value: Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
            attn_metadata: Metadata for the attention operation (unused).

        Returns:
            Output tensor of shape [batch_size, seq_len, num_heads, head_dim]
        """
        scale = self._get_scale_tensor(query.device)

        quant_q, q_descale = self._per_tensor_quant(
            query, scale=scale, quant_dtype=self._fp8_dtype, dtypeMax=self._dtype_max
        )
        quant_k, k_descale = self._per_tensor_quant(
            key, scale=scale, quant_dtype=self._fp8_dtype, dtypeMax=self._dtype_max
        )
        quant_v, v_descale = self._per_tensor_quant(
            value, scale=scale, quant_dtype=self._fp8_dtype, dtypeMax=self._dtype_max
        )

        kwargs = {}
        if self._has_descale:
            kwargs = {
                "q_descale": q_descale,
                "k_descale": k_descale,
                "v_descale": v_descale,
            }

        output = self._fp8_attn_fn(
            quant_q, quant_k, quant_v, causal=self.causal, **kwargs
        )
        return output
