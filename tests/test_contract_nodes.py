"""Workflow-compatibility contract: what saved workflows depend on.

Saved workflows reference nodes by class key, wire sockets by name/slot, and
store widget values positionally, so these must only ever grow at the end.
"""
from __future__ import annotations

import pytest

from contract_helpers import assert_saved_value_fits, input_names, map_saved_widgets

FP16_CHECKPOINT = "_h3\\minimax_h3_latent_upscaler_3d_fp16.safetensors"

# widgets_values copied from the user's saved "Daisaw H3 v2" workflows.
SAVED_3D_UPSCALER = [FP16_CHECKPOINT, "scale by multiplier", 2, 32, True, "cuda", "fp16", False]
SAVED_PROVIDER = [FP16_CHECKPOINT, "cuda", "fp16", False]


@pytest.fixture(scope="module")
def mappings(h3up):
    return h3up.NODE_CLASS_MAPPINGS


@pytest.mark.parametrize(
    ("key", "returns"),
    [
        ("MinimaxH3LatentUpscalerNode2D", ("LATENT",)),
        ("MinimaxH3LatentUpscaler3D", ("*",)),
        ("MinimaxH3LatentUpscaler3DProvider", ("H3_LATENT_UPSCALER",)),
        ("MinimaxH3LatentUpscaler3DRefineHandoff", ("LATENT",)),
    ],
)
def test_node_is_registered_with_stable_outputs(mappings, key, returns):
    assert key in mappings
    assert tuple(mappings[key].RETURN_TYPES) == returns


def test_3d_upscaler_inputs_keep_their_order(mappings):
    names = input_names(mappings["MinimaxH3LatentUpscaler3D"])
    assert names["required"][:8] == [
        "latent", "model_name", "mode", "align", "keep_proportion",
        "device", "precision", "offload_after_upscale",
    ]


def test_provider_inputs_keep_their_order(mappings):
    names = input_names(mappings["MinimaxH3LatentUpscaler3DProvider"])
    assert names["required"][:4] == ["model_name", "device", "precision", "offload_after_upscale"]


def test_refine_handoff_inputs_keep_their_order(mappings):
    names = input_names(mappings["MinimaxH3LatentUpscaler3DRefineHandoff"])
    assert names["required"][:17] == [
        "latent", "noise", "sampler", "sigmas", "model_name", "mode", "scale", "width",
        "height", "megapixels", "align", "keep_proportion", "lock_audio", "cfg",
        "device", "precision", "offload_after_upscale",
    ]
    assert names["optional"][:5] == ["audio_latent", "refine_state", "model", "positive", "negative"]


def test_2d_upscaler_inputs_keep_their_order(mappings):
    names = input_names(mappings["MinimaxH3LatentUpscalerNode2D"])
    assert names["required"][:5] == ["latent", "model_name", "scale", "device", "precision"]


@pytest.mark.parametrize(
    ("key", "saved", "expected_widgets"),
    [
        (
            "MinimaxH3LatentUpscaler3D",
            SAVED_3D_UPSCALER,
            ["model_name", "mode", "scale", "align", "keep_proportion", "device", "precision", "offload_after_upscale"],
        ),
        (
            "MinimaxH3LatentUpscaler3DProvider",
            SAVED_PROVIDER,
            ["model_name", "device", "precision", "offload_after_upscale"],
        ),
    ],
)
def test_saved_workflow_widgets_still_map_to_the_same_inputs(mappings, key, saved, expected_widgets):
    mapping = map_saved_widgets(mappings[key], saved)

    assert [name for name, _spec, _value in mapping] == expected_widgets
    assert len(mapping) == len(saved)
    for name, spec, value in mapping:
        assert_saved_value_fits(name, spec, value)
