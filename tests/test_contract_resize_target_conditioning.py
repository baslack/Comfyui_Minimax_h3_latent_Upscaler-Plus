"""Shared-provider contract: Resize Target Conditioning and the 3D node's learned_upscaler input."""
from __future__ import annotations

import pytest
import torch

from contract_helpers import input_names, map_saved_widgets


def _conditioning():
    keyframe = {"resolved_frame_index": 0, "latent": torch.randn(1, 24, 1, 4, 6)}
    ref = {"kind": "image", "latent_h": 4, "latent_w": 6, "latent": torch.randn(1, 24, 1, 4, 6)}
    meta = {"minimax_keyframes": [keyframe], "minimax_refs": [ref]}
    return [[torch.randn(1, 3, 8), meta]], keyframe, ref


@pytest.fixture
def provider(provider_module, tiny_checkpoint):
    return provider_module.H3LatentUpscalerProvider(model_name=tiny_checkpoint, device="cpu", precision="fp32")


def test_resize_node_is_registered_with_stable_sockets(h3up):
    node = h3up.NODE_CLASS_MAPPINGS["MinimaxH3ResizeTargetConditioning"]

    assert input_names(node) == {"required": ["conditioning", "latent"], "optional": ["learned_upscaler"]}
    assert tuple(node.RETURN_TYPES) == ("CONDITIONING",)
    assert map_saved_widgets(node, []) == []


def test_3d_node_learned_upscaler_is_an_appended_optional_socket(h3up):
    names = input_names(h3up.NODE_CLASS_MAPPINGS["MinimaxH3LatentUpscaler3D"])

    assert names["optional"] == ["learned_upscaler"]


def test_resize_matches_keyframes_to_the_latent_grid_and_leaves_refs_alone(h3up):
    node = h3up.NODE_CLASS_MAPPINGS["MinimaxH3ResizeTargetConditioning"]()
    conditioning, keyframe, ref = _conditioning()

    (out,) = node.resize(conditioning, {"samples": torch.randn(1, 24, 7, 8, 12)})

    meta = out[0][1]
    assert meta["minimax_keyframes"][0]["latent"].shape == (1, 24, 1, 8, 12)
    assert meta["minimax_refs"][0] is ref
    assert keyframe["latent"].shape == (1, 24, 1, 4, 6)


def test_resize_with_provider_uses_the_learned_upscale(h3up, provider):
    node = h3up.NODE_CLASS_MAPPINGS["MinimaxH3ResizeTargetConditioning"]()
    conditioning, keyframe, _ref = _conditioning()

    (out,) = node.resize(conditioning, {"samples": torch.randn(1, 24, 7, 8, 12)}, learned_upscaler=provider)

    expected = provider.upscale_clean_video(keyframe["latent"], target_h=8, target_w=12)
    assert torch.equal(out[0][1]["minimax_keyframes"][0]["latent"], expected)


def test_3d_node_with_provider_matches_the_node_with_the_same_checkpoint(h3up, provider, tiny_checkpoint):
    node = h3up.NODE_CLASS_MAPPINGS["MinimaxH3LatentUpscaler3D"]
    video = torch.randn(1, 24, 7, 4, 6)
    kwargs = dict(
        latent={"samples": video}, model_name=tiny_checkpoint,
        mode={"mode": "scale by multiplier", "scale": 2.0}, align=16, keep_proportion=True,
        device="cpu", precision="fp32", offload_after_upscale=False,
    )

    shared = node.execute(learned_upscaler=provider, **kwargs).result[0]["samples"]
    own = node.execute(**kwargs).result[0]["samples"]

    assert shared.shape == (1, 24, 7, 8, 12)
    assert torch.equal(shared, own)
