"""Split/Combine AV Latent contract: schema, lossless round trip, clear errors."""
from __future__ import annotations

import pytest
import torch

from contract_helpers import input_names


@pytest.fixture
def joint_latent():
    import comfy.nested_tensor

    video = torch.randn(1, 24, 7, 4, 6)
    audio = torch.randn(1, 32, 2, 40)
    return video, audio, {"samples": comfy.nested_tensor.NestedTensor([video, audio]), "noise_mask": torch.ones(1)}


def test_split_and_combine_are_registered_with_stable_sockets(h3up):
    split = h3up.NODE_CLASS_MAPPINGS["MinimaxH3SplitAVLatent"]
    combine = h3up.NODE_CLASS_MAPPINGS["MinimaxH3CombineAVLatent"]

    assert input_names(split)["required"] == ["latent"]
    assert tuple(split.RETURN_TYPES) == ("LATENT", "LATENT")
    assert input_names(combine)["required"] == ["video_latent", "audio_latent"]
    assert tuple(combine.RETURN_TYPES) == ("LATENT",)


def test_split_then_combine_round_trips_both_streams_exactly(h3up, joint_latent):
    video, audio, latent = joint_latent
    split = h3up.NODE_CLASS_MAPPINGS["MinimaxH3SplitAVLatent"]()
    combine = h3up.NODE_CLASS_MAPPINGS["MinimaxH3CombineAVLatent"]()

    video_latent, audio_latent = split.split(latent)
    (joint,) = combine.combine(video_latent, audio_latent)

    assert torch.equal(video_latent["samples"], video)
    assert torch.equal(audio_latent["samples"], audio)
    assert "noise_mask" not in video_latent and "noise_mask" not in audio_latent
    rebuilt_video, rebuilt_audio = joint["samples"].unbind()
    assert torch.equal(rebuilt_video, video)
    assert torch.equal(rebuilt_audio, audio)


def test_split_rejects_a_plain_latent(h3up):
    split = h3up.NODE_CLASS_MAPPINGS["MinimaxH3SplitAVLatent"]()

    with pytest.raises(ValueError, match="nothing to split"):
        split.split({"samples": torch.randn(1, 24, 2, 4, 4)})


def test_combine_rejects_swapped_streams(h3up, joint_latent):
    video, audio, _latent = joint_latent
    combine = h3up.NODE_CLASS_MAPPINGS["MinimaxH3CombineAVLatent"]()

    with pytest.raises(ValueError):
        combine.combine({"samples": audio}, {"samples": video})
