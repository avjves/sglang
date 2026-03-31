# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class AITERSageV2Backend(AttentionBackend):

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER_SAGE_V2

    @staticmethod
    def get_impl_cls() -> type["AITERSageV2Impl"]:
        return AITERSageV2Impl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError(
            "AITER Sage V2 backend does not have a metadata builder."
        )


class AITERSageV2Impl(AttentionImpl):

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

        try:
            from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4 import (
                create_hadamard_matrix,
            )
            from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
                fav3_sage_mxfp4_wrapper,
            )
        except ImportError:
            raise ImportError(
                "AITER Sage V2 attention is not available, please update AITER version."
            )

        self.attn_fn = fav3_sage_mxfp4_wrapper

        block_r = 128
        hadamard = create_hadamard_matrix(block_r, dtype=torch.bfloat16) / (
            block_r**0.5
        )

        # Replicate Hadamard matrix on each available GPU
        self._hadamard: dict[torch.device, torch.Tensor] = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f"cuda:{i}")
                self._hadamard[device] = hadamard.to(device)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """
        Performs attention using AITER Sage V2 (MXFP4 + Hadamard rotation).

        Args:
            query: Query tensor of shape [batch_size, seq_len, head_num, head_dim]
            key: Key tensor of shape [batch_size, seq_len, head_num, head_dim]
            value: Value tensor of shape [batch_size, seq_len, head_num, head_dim]
            attn_metadata: Metadata for the attention operation (unused).

        Returns:
            Output tensor of shape [batch_size, seq_len, head_num, head_dim]
        """
        # Contiguous is needed for Sage V2 in older AITER versions.
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        output = self.attn_fn(
            query,
            key,
            value,
            hadamard_rotation=True,
            R=self._hadamard[query.device],
            causal=self.causal,
        )
        return output
