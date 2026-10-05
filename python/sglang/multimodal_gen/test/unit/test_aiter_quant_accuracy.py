# SPDX-License-Identifier: Apache-2.0
"""Each aiter_quant format against an SDPA reference (ROCm-only; skipped elsewhere).

The budgets below are deliberately loose: they are not a precision claim for any
format, they are the line between "lossy as designed" and "wrong". A row wired
to a format pair aiter quantizes differently than the backend assumed -- the V
operand packed in the FP6-P order when the kernel reads it dense, say -- returns
a plausible-looking tensor whose cosine against the reference is near zero, and
that is the failure this catches without a GPU-hours calibration run.

Tighten a budget once the format has been measured on the target part; the
measured values print with `-s`.
"""

import pytest
import torch

from sglang.multimodal_gen.runtime.server_args import get_global_server_args

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip),
    reason="aiter_quant is a ROCm backend",
)

HEADS = 8
HEAD_DIM = 128
SEQUENCE = 2048
SOFTMAX_SCALE = HEAD_DIM**-0.5

# (relative RMS ceiling, cosine floor) per format, ordered by how much of the
# operand each one throws away. Provisional -- see the module docstring.
_BUDGETS = {
    "bf16fp8": (0.06, 0.998),  # Q/K stay BF16; only V is quantized.
    "mxfp8": (0.12, 0.990),  # FP8 operands, E8M0 scales per 32 elements.
    "fp8": (0.15, 0.985),  # FP8 operands, one scale per tensor.
    "i8fp8": (0.15, 0.985),  # INT8 Q/K after the shared-K subtraction.
    "mxfp6": (0.30, 0.960),  # 3 mantissa bits on Q/K.
    "mxfp4": (0.60, 0.880),  # 1 mantissa bit, on Q/K and V both.
}


def _backend():
    # `aiter` ships with ROCm only, and mha_v4 needs a recent build.
    pytest.importorskip("aiter", reason="AITer is a ROCm-only dependency")
    pytest.importorskip("aiter.ops.mha_v4", reason="this aiter build has no mha_v4")
    from sglang.multimodal_gen.runtime.layers.attention.backends import aiter_quant

    return aiter_quant


def _build_impl(aiter_quant, format_name: str):
    """Build the impl the way a run does, through --attention-backend-config.

    The conftest fixture hands each test its own server args, so the selection
    is undone when the test ends.
    """
    get_global_server_args().attention_backend_config = {"format": format_name}
    try:
        return aiter_quant.AITERQuantImpl(
            num_heads=HEADS, head_size=HEAD_DIM, softmax_scale=SOFTMAX_SCALE
        )
    except NotImplementedError as missing_row:
        # gfx942 carries only a subset of the rows.
        pytest.skip(str(missing_row))


def _qkv(kv_sequence: int = SEQUENCE, kv_heads: int = HEADS):
    generator = torch.Generator(device="cuda").manual_seed(0)

    def draw(sequence, heads):
        return torch.randn(
            (1, sequence, heads, HEAD_DIM),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )

    return (
        draw(SEQUENCE, HEADS),
        draw(kv_sequence, kv_heads),
        draw(kv_sequence, kv_heads),
    )


def _reference(query, key, value):
    """Non-causal SDPA over the same BSHD operands, in BHSD and unquantized."""
    out = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        scale=SOFTMAX_SCALE,
        enable_gqa=key.shape[2] != query.shape[2],
    )
    return out.transpose(1, 2)


def _error_metrics(candidate, reference):
    candidate = candidate.float()
    reference = reference.float()
    relative_rms = (candidate - reference).pow(2).mean().sqrt() / reference.pow(
        2
    ).mean().sqrt()
    cosine = torch.nn.functional.cosine_similarity(
        candidate.flatten(), reference.flatten(), dim=0
    )
    return float(relative_rms), float(cosine)


@pytest.mark.parametrize("format_name", sorted(_BUDGETS))
def test_format_tracks_the_sdpa_reference(format_name):
    aiter_quant = _backend()
    impl = _build_impl(aiter_quant, format_name)
    query, key, value = _qkv()

    output = impl.forward(query, key, value, None)
    expected = _reference(query, key, value)

    assert output.shape == expected.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()

    relative_rms, cosine = _error_metrics(output, expected)
    print(f"\n{format_name}: relative_rms={relative_rms:.4f} cosine={cosine:.5f}")
    rms_ceiling, cosine_floor = _BUDGETS[format_name]
    assert relative_rms < rms_ceiling, f"{format_name}: {relative_rms}"
    assert cosine > cosine_floor, f"{format_name}: {cosine}"


@pytest.mark.parametrize("format_name", sorted(_BUDGETS))
def test_format_handles_cross_attention_and_gqa(format_name):
    # Cross attention gives K/V their own sequence length, and the DiT blocks
    # that use it may also send fewer KV heads than query heads. Both travel
    # through the same quantizer, so each row has to accept them.
    aiter_quant = _backend()
    impl = _build_impl(aiter_quant, format_name)
    query, key, value = _qkv(kv_sequence=512, kv_heads=HEADS // 4)

    output = impl.forward(query, key, value, None)
    expected = _reference(query, key, value)

    assert output.shape == query.shape
    assert torch.isfinite(output.float()).all()
    relative_rms, cosine = _error_metrics(output, expected)
    print(f"\n{format_name} (cross/GQA): rms={relative_rms:.4f} cos={cosine:.5f}")
    assert cosine > _BUDGETS[format_name][1]


def test_budgets_cover_every_format():
    # A new row in the backend table must arrive with a budget, or it ships
    # with no accuracy test at all and nothing here says so.
    aiter_quant = _backend()
    assert sorted(_BUDGETS) == sorted(fmt.name for fmt in aiter_quant._FORMATS)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
