# SPDX-License-Identifier: Apache-2.0
#
# Quantized AITER attention family backend (ROCm / gfx950).
#
# One backend, several quant formats selected via --attention-backend-config
# (e.g. `format=mxfp4`). All formats share the same skeleton --
# (optional Hadamard rotation) -> quantize Q/K/V -> aiter kernel -- and differ
# only in the per-Q/K/V quant op, the format enums passed to the kernel, and the
# softmax-scale handling.
#
# `format=fp8` (the default) uses aiter.flash_attn_fp8_pertensor_func with
# Hadamard-rotated per-tensor quantization; the other five formats
# (i8fp8, mxfp4, mxfp6, f4f4, f6f4) funnel into aiter.ops.mha_v4.mha_v4_packed.
#
# The quant math is ported from xDiT's mxfp_fmha_asm branch
# (xfuser/core/distributed/attention_backend.py). Unlike xDiT -- whose attention
# functions receive a [batch, heads, seq, dim] layout and permute to
# [batch, seq, heads, dim] for the aiter kernel -- SGL-D's AttentionImpl.forward
# already receives [batch, seq, heads, dim] (the layout the kernel expects), so
# xDiT's permutes are dropped here.

import inspect
from collections.abc import Callable

import aiter
import msgspec
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.utils import is_gfx95_supported

logger = init_logger(__name__)

# All aiter_quant kernels target full-MHA head_dim==128 models (e.g. Wan
# self-/cross-attention) on a gfx950-class arch. Selecting this backend is
# explicit, so unmet constraints raise rather than silently falling back.
_REQUIRED_HEAD_DIM = 128

# Hadamard block size for the fp8 path. Matches xDiT's hardcoded 128 (full-head
# rotation for head_dim==128).
_HADAMARD_BLOCK_R = 128

_DEFAULT_FORMAT = "fp8"


# ---------------------------------------------------------------------------
# aiter.ops.mha_v4 imports (required by every format except fp8).
#
# Imported at module load so torch.compile sees stable symbols. If the installed
# aiter lacks mha_v4, the names resolve to None and construction of a non-fp8
# format raises a clear error.
# ---------------------------------------------------------------------------
try:
    from aiter.ops.mha_v4 import (
        AttentionFormat as _AiterAttentionFormat,
        mha_v4_packed as _aiter_mha_v4_packed,
        mha_v4_q_multiplier as _aiter_mha_v4_q_multiplier,
        mxfp4_k_view as _aiter_mxfp4_k_view,
        mxfp4_v_view as _aiter_mxfp4_v_view,
        mxfp6_k_view as _aiter_mxfp6_k_view,
        native_fp8_format as _aiter_native_fp8_format,
        quantize_fp8 as _aiter_mha_v4_quantize_fp8,
        quantize_int8 as _aiter_mha_v4_quantize_int8,
        quantize_mxfp4_k as _aiter_quantize_mxfp4_k,
        quantize_mxfp4_q as _aiter_quantize_mxfp4_q,
        quantize_mxfp6_k as _aiter_quantize_mxfp6_k,
        quantize_mxfp6_q as _aiter_quantize_mxfp6_q,
        quantize_v_fp8 as _aiter_quantize_v_fp8,
        quantize_v_mxfp4 as _aiter_quantize_v_mxfp4,
        scale_modes_for_formats as _aiter_scale_modes_for_formats,
    )

    _AITER_MHA_V4_AVAILABLE = True
except ImportError:
    # Keep the names defined (as None) so they remain patchable and referencing
    # them yields a clear message via the construction-time availability check.
    _AiterAttentionFormat = None
    _aiter_mha_v4_packed = None
    _aiter_mha_v4_q_multiplier = None
    _aiter_mxfp4_k_view = None
    _aiter_mxfp4_v_view = None
    _aiter_mxfp6_k_view = None
    _aiter_native_fp8_format = None
    _aiter_mha_v4_quantize_fp8 = None
    _aiter_mha_v4_quantize_int8 = None
    _aiter_quantize_mxfp4_k = None
    _aiter_quantize_mxfp4_q = None
    _aiter_quantize_mxfp6_k = None
    _aiter_quantize_mxfp6_q = None
    _aiter_quantize_v_fp8 = None
    _aiter_quantize_v_mxfp4 = None
    _aiter_scale_modes_for_formats = None

    _AITER_MHA_V4_AVAILABLE = False


# ---------------------------------------------------------------------------
# Hadamard rotation helpers (fp8 path only).
# ---------------------------------------------------------------------------
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
    except (AttributeError, TypeError, ValueError):
        return False


AITER_FP8_HAS_DESCALE = _aiter_fp8_has_descale()


# ---------------------------------------------------------------------------
# Per-format forward pipelines. Each receives [batch, seq, heads, dim] tensors
# and returns [batch, seq, heads, dim]. These are plain functions (no
# torch.compiler.disable / custom-op wrapper), mirroring xDiT's aiter quant
# attention calls: the aiter quant op and kernel are invoked inline, so torch
# handles the aiter ops at their own boundaries rather than us forcing a graph
# break around the whole pipeline.
# ---------------------------------------------------------------------------
def _forward_fp8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """fp8 per-tensor quantization + Hadamard-rotated Q/K via
    aiter.flash_attn_fp8_pertensor_func."""
    R = FP8_HADAMARD_MATRIX[query.device]
    # Rotate Q and K only; V is quantized but not rotated. Q@K^T is preserved
    # because R @ R.T == I.
    query = _fp8_hadamard_rotate(query, R).contiguous()
    key = _fp8_hadamard_rotate(key, R).contiguous()
    value = value.contiguous()

    quant_dtype = aiter.dtypes.fp8
    dtype_max = torch.finfo(quant_dtype).max
    # Dynamic per-tensor scale when descale vectors are supported, else a static
    # scale of 1.0 (no descale) -- matches xDiT's default.
    scale = None
    if not AITER_FP8_HAS_DESCALE:
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
        causal=causal,
        softmax_scale=softmax_scale,
        **descale_kwargs,
    )


def _forward_i8fp8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """int8 Q/K + fp8 V via mha_v4_packed (no Hadamard rotation)."""
    if causal:
        raise NotImplementedError(
            "aiter_quant i8fp8 does not support causal masking."
        )
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    q_i8, q_descale = _aiter_mha_v4_quantize_int8(query, 1.0)
    k_i8, k_descale = _aiter_mha_v4_quantize_int8(key, 1.0)
    v_fp8, v_descale = _aiter_mha_v4_quantize_fp8(value)

    fp8_format = _aiter_native_fp8_format()
    return _aiter_mha_v4_packed(
        q_i8,
        k_i8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
        _AiterAttentionFormat.INT8,
        _AiterAttentionFormat.INT8,
        fp8_format,
        *_aiter_scale_modes_for_formats(
            _AiterAttentionFormat.INT8,
            _AiterAttentionFormat.INT8,
            fp8_format,
        ),
    )


def _forward_mxfp4(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """fp4 (E2M1) Q/K + fp8 V via mha_v4_packed. Hadamard rotation is fused into
    the fp4 quant op; softmax_scale is baked into the Q multiplier."""
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # The mxfp4 ASM kernel expects head_dim**-0.5 baked into the Q multiplier.
    softmax_scale = query.shape[-1] ** -0.5

    q_fp4, q_scale = _aiter_quantize_mxfp4_q(
        query, _aiter_mha_v4_q_multiplier(softmax_scale)
    )
    k_buf, k_scale = _aiter_quantize_mxfp4_k(key)
    v_fp8, v_scale = _aiter_quantize_v_fp8(value)

    k_fp4 = _aiter_mxfp4_k_view(k_buf, k_scale)
    fp8_format = _aiter_native_fp8_format()
    return _aiter_mha_v4_packed(
        q_fp4,
        k_fp4,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        _AiterAttentionFormat.MXFP4,
        _AiterAttentionFormat.MXFP4,
        fp8_format,
        *_aiter_scale_modes_for_formats(
            _AiterAttentionFormat.MXFP4,
            _AiterAttentionFormat.MXFP4,
            fp8_format,
        ),
        softmax_scale=softmax_scale,
    )


def _forward_f4f4(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """true-MXFP4 Q/K/V via mha_v4_packed."""
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    softmax_scale = query.shape[-1] ** -0.5

    q_fp4, q_scale = _aiter_quantize_mxfp4_q(
        query, _aiter_mha_v4_q_multiplier(softmax_scale)
    )
    k_buf, k_scale = _aiter_quantize_mxfp4_k(key)
    v_buf, v_scale = _aiter_quantize_v_mxfp4(value)

    kv_len = value.shape[1]
    v_fp4 = _aiter_mxfp4_v_view(v_buf, v_scale, kv_len)
    k_fp4 = _aiter_mxfp4_k_view(k_buf, k_scale)
    return _aiter_mha_v4_packed(
        q_fp4,
        k_fp4,
        v_fp4,
        q_scale,
        k_scale,
        v_scale,
        _AiterAttentionFormat.MXFP4,
        _AiterAttentionFormat.MXFP4,
        _AiterAttentionFormat.MXFP4,
        *_aiter_scale_modes_for_formats(
            _AiterAttentionFormat.MXFP4,
            _AiterAttentionFormat.MXFP4,
            _AiterAttentionFormat.MXFP4,
        ),
        softmax_scale=softmax_scale,
    )


def _forward_mxfp6(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """fp6 (E2M3) Q/K + fp8 V via mha_v4_packed."""
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    softmax_scale = query.shape[-1] ** -0.5

    q_fp6, q_scale = _aiter_quantize_mxfp6_q(
        query, _aiter_mha_v4_q_multiplier(softmax_scale)
    )
    k_buf, k_scale_buf = _aiter_quantize_mxfp6_k(key)
    v_fp8, v_scale = _aiter_quantize_v_fp8(value)

    b, s, h, _ = v_fp8.shape
    k_fp6, k_scale = _aiter_mxfp6_k_view(k_buf, k_scale_buf, b, s, h)
    fp8_format = _aiter_native_fp8_format()
    return _aiter_mha_v4_packed(
        q_fp6,
        k_fp6,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        _AiterAttentionFormat.MXFP6,
        _AiterAttentionFormat.MXFP6,
        fp8_format,
        *_aiter_scale_modes_for_formats(
            _AiterAttentionFormat.MXFP6,
            _AiterAttentionFormat.MXFP6,
            fp8_format,
        ),
        softmax_scale=softmax_scale,
    )


def _forward_f6f4(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """MXFP6 Q/K + true-MXFP4 V via mha_v4_packed."""
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    softmax_scale = query.shape[-1] ** -0.5

    q_fp6, q_scale = _aiter_quantize_mxfp6_q(
        query, _aiter_mha_v4_q_multiplier(softmax_scale)
    )
    k_buf, k_scale_buf = _aiter_quantize_mxfp6_k(key)
    v_buf, v_scale = _aiter_quantize_v_mxfp4(value)

    kv_len = value.shape[1]
    v_fp4 = _aiter_mxfp4_v_view(v_buf, v_scale, kv_len)
    b, _, h, _ = v_fp4.shape
    k_fp6, k_scale = _aiter_mxfp6_k_view(k_buf, k_scale_buf, b, kv_len, h)
    return _aiter_mha_v4_packed(
        q_fp6,
        k_fp6,
        v_fp4,
        q_scale,
        k_scale,
        v_scale,
        _AiterAttentionFormat.MXFP6,
        _AiterAttentionFormat.MXFP6,
        _AiterAttentionFormat.MXFP4,
        *_aiter_scale_modes_for_formats(
            _AiterAttentionFormat.MXFP6,
            _AiterAttentionFormat.MXFP6,
            _AiterAttentionFormat.MXFP4,
        ),
        softmax_scale=softmax_scale,
    )


class _FormatSpec(msgspec.Struct, frozen=True):
    """A single aiter_quant variant: its forward pipeline and whether it needs
    the aiter.ops.mha_v4 kernels (all but fp8 do)."""

    forward: Callable[..., torch.Tensor]
    requires_mha_v4: bool


_FORMATS: dict[str, _FormatSpec] = {
    "fp8": _FormatSpec(forward=_forward_fp8, requires_mha_v4=False),
    "i8fp8": _FormatSpec(forward=_forward_i8fp8, requires_mha_v4=True),
    "mxfp4": _FormatSpec(forward=_forward_mxfp4, requires_mha_v4=True),
    "mxfp6": _FormatSpec(forward=_forward_mxfp6, requires_mha_v4=True),
    "f4f4": _FormatSpec(forward=_forward_f4f4, requires_mha_v4=True),
    "f6f4": _FormatSpec(forward=_forward_f6f4, requires_mha_v4=True),
}


def _resolve_format() -> str:
    """Read the `format` key from --attention-backend-config (default fp8)."""
    cfg = get_global_server_args().attention_backend_config or {}
    return str(cfg.get("format", _DEFAULT_FORMAT)).lower()


class AITerQuantBackend(AttentionBackend):
    """AITER quantized attention family backend (ROCm)."""

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER_QUANT

    @staticmethod
    def get_impl_cls() -> type["AITerQuantImpl"]:
        return AITerQuantImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        # AITER quant backend does not require special metadata.
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError(
            "AITER quant backend does not have a metadata builder."
        )


class AITerQuantImpl(AttentionImpl):
    """Quantized attention via aiter, with the variant selected by the `format`
    key of --attention-backend-config (fp8, i8fp8, mxfp4, mxfp6, f4f4, f6f4)."""

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
        fmt = _resolve_format()
        spec = _FORMATS.get(fmt)
        if spec is None:
            raise ValueError(
                f"Unknown aiter_quant format {fmt!r}. Set "
                "--attention-backend-config format=<name> to one of: "
                f"{', '.join(sorted(_FORMATS))}."
            )

        if num_kv_heads is not None and num_kv_heads != num_heads:
            raise NotImplementedError(
                "AITER quant backend does not support Grouped Query Attention "
                f"(num_heads={num_heads}, num_kv_heads={num_kv_heads})."
            )
        if head_size != _REQUIRED_HEAD_DIM:
            raise NotImplementedError(
                f"AITER quant backend requires head_dim == {_REQUIRED_HEAD_DIM}, "
                f"got {head_size}."
            )
        if not is_gfx95_supported():
            raise RuntimeError(
                "AITER quant backend requires a gfx950-class arch."
            )
        if spec.requires_mha_v4 and not _AITER_MHA_V4_AVAILABLE:
            raise RuntimeError(
                f"aiter_quant format {fmt!r} requires aiter.ops.mha_v4, which is "
                "not available in the installed aiter build."
            )

        self.format = fmt
        self._spec = spec
        self.causal = causal
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale

        # Deduped per message, so this logs once per format for the whole run.
        logger.info_once(f"aiter_quant attention backend using format={fmt}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """
        Performs quantized attention using the configured format.

        Args:
            query: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            key: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            value: Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
            attn_metadata: Metadata for the attention operation (unused).

        Returns:
            Output tensor of shape [batch_size, seq_len, num_heads, head_dim]
        """
        return self._spec.forward(
            query,
            key,
            value,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
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
            "AITER quant backend does not support varlen attention."
        )
