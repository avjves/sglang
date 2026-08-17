"""Unit tests for the AITER quant family attention backend.

`aiter` is not importable in CI, so it is stubbed in sys.modules before the
backend module is imported. The aiter.ops.mha_v4 kernels are then monkeypatched
per-test to assert each format routes to the expected quant ops / format enums.
"""

import collections
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

# Stub `aiter` before importing the backend (it does `import aiter` at module
# top). A plain MagicMock has no package __path__, so `from aiter.ops.mha_v4
# import ...` fails -> the backend loads with mha_v4 marked unavailable.
sys.modules.setdefault("aiter", MagicMock())

from sglang.multimodal_gen.runtime.layers.attention.backends import (  # noqa: E402
    aiter_quant,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.aiter_quant import (  # noqa: E402
    AITerQuantBackend,
    AITerQuantImpl,
    _FORMATS,
)
from sglang.multimodal_gen.runtime.platforms.interface import (  # noqa: E402
    AttentionBackendEnum,
)

_HEAD_DIM = 128


class _Fmt:
    """Sentinel AttentionFormat enum stand-in."""

    INT8 = "INT8"
    MXFP4 = "MXFP4"
    MXFP6 = "MXFP6"


def _pair(x, *_a, **_k):
    """Fake quant op: returns (tensor, scale)."""
    return x, torch.ones(1)


def _server_args(cfg: dict) -> MagicMock:
    sa = MagicMock()
    sa.attention_backend_config = cfg
    return sa


def _construct(fmt: str, *, head_size: int = _HEAD_DIM, num_kv_heads=None,
               gfx95: bool = True, mha_v4: bool = True) -> AITerQuantImpl:
    with (
        patch.object(aiter_quant, "is_gfx95_supported", lambda: gfx95),
        patch.object(aiter_quant, "_AITER_MHA_V4_AVAILABLE", mha_v4),
        patch.object(
            aiter_quant, "get_global_server_args",
            return_value=_server_args({"format": fmt}),
        ),
    ):
        return AITerQuantImpl(
            num_heads=2,
            head_size=head_size,
            softmax_scale=head_size**-0.5,
            causal=False,
            num_kv_heads=num_kv_heads,
        )


class TestAITerQuantBackend(unittest.TestCase):
    def test_enum_name(self):
        self.assertEqual(str(AttentionBackendEnum.AITER_QUANT), "aiter_quant")

    def test_backend_wiring(self):
        self.assertIs(AITerQuantBackend.get_impl_cls(), AITerQuantImpl)
        self.assertEqual(
            AITerQuantBackend.get_enum(), AttentionBackendEnum.AITER_QUANT
        )

    def test_dispatch_has_all_formats(self):
        self.assertEqual(
            set(_FORMATS),
            {"fp8", "i8fp8", "mxfp4", "mxfp6", "f4f4", "f6f4"},
        )
        # Only fp8 avoids the mha_v4 kernels.
        self.assertFalse(_FORMATS["fp8"].requires_mha_v4)
        for fmt in ("i8fp8", "mxfp4", "mxfp6", "f4f4", "f6f4"):
            self.assertTrue(_FORMATS[fmt].requires_mha_v4)

    def test_default_format_is_fp8(self):
        with (
            patch.object(aiter_quant, "is_gfx95_supported", lambda: True),
            patch.object(
                aiter_quant, "get_global_server_args",
                return_value=_server_args({}),
            ),
        ):
            impl = AITerQuantImpl(
                num_heads=2, head_size=_HEAD_DIM, softmax_scale=_HEAD_DIM**-0.5
            )
        self.assertEqual(impl.format, "fp8")

    def test_format_parsing(self):
        self.assertEqual(_construct("mxfp4").format, "mxfp4")

    def test_invalid_format_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _construct("bogus")
        msg = str(ctx.exception)
        for fmt in _FORMATS:
            self.assertIn(fmt, msg)

    def test_gqa_guard(self):
        with self.assertRaises(NotImplementedError):
            _construct("fp8", num_kv_heads=1)

    def test_head_dim_guard(self):
        with self.assertRaises(NotImplementedError):
            _construct("fp8", head_size=64)

    def test_gfx95_guard(self):
        with self.assertRaises(RuntimeError):
            _construct("fp8", gfx95=False)

    def test_mha_v4_unavailable_guard(self):
        # fp8 does not need mha_v4, so it still constructs.
        self.assertEqual(_construct("fp8", mha_v4=False).format, "fp8")
        # The five quant formats require it.
        for fmt in ("i8fp8", "mxfp4", "mxfp6", "f4f4", "f6f4"):
            with self.assertRaises(RuntimeError):
                _construct(fmt, mha_v4=False)

    def _run_mha_v4_format(self, fmt: str):
        """Route `fmt` through a fully faked mha_v4 and return the recorded
        mha_v4_packed call."""
        recorder = MagicMock(return_value=torch.zeros(1, 8, 2, _HEAD_DIM))
        overrides = dict(
            _AiterAttentionFormat=_Fmt,
            _aiter_mha_v4_packed=recorder,
            _aiter_mha_v4_q_multiplier=lambda s: 1.0,
            _aiter_native_fp8_format=lambda: "FP8",
            _aiter_scale_modes_for_formats=lambda *a: (),
            _aiter_mha_v4_quantize_int8=_pair,
            _aiter_mha_v4_quantize_fp8=_pair,
            _aiter_quantize_mxfp4_q=_pair,
            _aiter_quantize_mxfp4_k=_pair,
            _aiter_quantize_mxfp6_q=_pair,
            _aiter_quantize_mxfp6_k=_pair,
            _aiter_quantize_v_fp8=_pair,
            _aiter_quantize_v_mxfp4=_pair,
            _aiter_mxfp4_k_view=lambda buf, scale: buf,
            _aiter_mxfp4_v_view=lambda buf, scale, kv_len: buf,
            _aiter_mxfp6_k_view=lambda buf, scale, b, s, h: (buf, scale),
        )
        impl = _construct(fmt)
        q = k = v = torch.zeros(1, 8, 2, _HEAD_DIM, dtype=torch.bfloat16)
        with patch.multiple(aiter_quant, **overrides):
            impl.forward(q, k, v)
        recorder.assert_called_once()
        return recorder.call_args

    def test_route_i8fp8(self):
        call = self._run_mha_v4_format("i8fp8")
        self.assertEqual(call.args[6:9], (_Fmt.INT8, _Fmt.INT8, "FP8"))
        # i8fp8 leaves softmax_scale to the kernel default.
        self.assertNotIn("softmax_scale", call.kwargs)

    def test_route_mxfp4(self):
        call = self._run_mha_v4_format("mxfp4")
        self.assertEqual(call.args[6:9], (_Fmt.MXFP4, _Fmt.MXFP4, "FP8"))
        self.assertAlmostEqual(call.kwargs["softmax_scale"], _HEAD_DIM**-0.5)

    def test_route_mxfp6(self):
        call = self._run_mha_v4_format("mxfp6")
        self.assertEqual(call.args[6:9], (_Fmt.MXFP6, _Fmt.MXFP6, "FP8"))
        self.assertAlmostEqual(call.kwargs["softmax_scale"], _HEAD_DIM**-0.5)

    def test_route_f4f4(self):
        call = self._run_mha_v4_format("f4f4")
        self.assertEqual(call.args[6:9], (_Fmt.MXFP4, _Fmt.MXFP4, _Fmt.MXFP4))
        self.assertAlmostEqual(call.kwargs["softmax_scale"], _HEAD_DIM**-0.5)

    def test_route_f6f4(self):
        call = self._run_mha_v4_format("f6f4")
        self.assertEqual(call.args[6:9], (_Fmt.MXFP6, _Fmt.MXFP6, _Fmt.MXFP4))
        self.assertAlmostEqual(call.kwargs["softmax_scale"], _HEAD_DIM**-0.5)

    def test_route_fp8(self):
        impl = _construct("fp8")
        fake_aiter = types.SimpleNamespace(
            dtypes=types.SimpleNamespace(fp8=torch.float8_e4m3fn),
            per_tensor_quant=lambda x, **_k: (x, torch.ones(1)),
            flash_attn_fp8_pertensor_func=MagicMock(
                return_value=torch.zeros(1, 8, 2, _HEAD_DIM)
            ),
        )
        q = k = v = torch.zeros(1, 8, 2, _HEAD_DIM, dtype=torch.bfloat16)
        # Skip the Hadamard rotation (its per-device matrix may not carry a CPU
        # entry on a GPU host); R=None makes _fp8_hadamard_rotate a no-op.
        no_rotate = collections.defaultdict(lambda: None)
        with (
            patch.object(aiter_quant, "aiter", fake_aiter),
            patch.object(aiter_quant, "AITER_FP8_HAS_DESCALE", False),
            patch.object(aiter_quant, "FP8_HADAMARD_MATRIX", no_rotate),
        ):
            impl.forward(q, k, v)
        fake_aiter.flash_attn_fp8_pertensor_func.assert_called_once()
        kwargs = fake_aiter.flash_attn_fp8_pertensor_func.call_args.kwargs
        self.assertFalse(kwargs["causal"])
        self.assertAlmostEqual(kwargs["softmax_scale"], _HEAD_DIM**-0.5)

    def test_forward_varlen_not_implemented(self):
        impl = _construct("fp8")
        with self.assertRaises(NotImplementedError):
            impl.forward_varlen(
                torch.zeros(1, 8, 2, _HEAD_DIM),
                torch.zeros(1, 8, 2, _HEAD_DIM),
                torch.zeros(1, 8, 2, _HEAD_DIM),
                cu_seqlens=torch.zeros(2, dtype=torch.int32),
                max_seqlen=8,
            )


if __name__ == "__main__":
    unittest.main()
