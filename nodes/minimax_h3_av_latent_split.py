"""Split/combine the joint MiniMax H3 AV LATENT.

The native MiniMax H3 AV latent is a two-member ``NestedTensor([video, audio])``
stored in ``LATENT["samples"]``. Most graphs never need to see that directly --
the integrated refine node accepts it as one connection -- but a plain
video-only node (such as the standalone learned upscaler) cannot: it expects
``samples`` to be a bare tensor and has no way to carry the audio member
through untouched. These two utility nodes expose the split/combine step
directly, mirroring the LATENT split/combine utilities other AV model
integrations (e.g. LTX) provide, so a video-only node can run on the video
member alone and the result can be reassembled afterward without a second H3
sampling pass.
"""
from __future__ import annotations

from .minimax_h3_refine_support import (
    _is_nested_tensor,
    _plain_latent_samples,
    _validate_audio,
    _validate_h3_av_samples,
    _validate_video,
    _wrap_h3_samples,
)


class MinimaxH3SplitAVLatent:
    """Split a joint MiniMax H3 AV LATENT into separate video/audio LATENTs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent")
    FUNCTION = "split"
    CATEGORY = "video/MinimaxH3"
    DESCRIPTION = (
        "Splits a joint MiniMax H3 AV LATENT (NestedTensor[video, audio]) into a "
        "plain 24-channel video LATENT and a plain 32-channel audio LATENT. Use "
        "this to run video-only nodes (e.g. the standalone learned upscaler) on "
        "just the video member, then reassemble with 'MiniMax H3 Combine AV "
        "Latent'. Any existing noise_mask is dropped on both outputs since a "
        "joint AV mask does not apply to a single split stream."
    )

    def split(self, latent):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("latent must be a LATENT dictionary containing 'samples'")
        samples = latent["samples"]
        if not _is_nested_tensor(samples):
            raise ValueError(
                "latent does not contain a joint MiniMax H3 AV NestedTensor "
                "([video, audio]) -- there is nothing to split; this is already "
                "a plain LATENT."
            )
        video, audio = _validate_h3_av_samples(samples)

        video_latent = dict(latent)
        video_latent["samples"] = video
        video_latent.pop("noise_mask", None)

        audio_latent = dict(latent)
        audio_latent["samples"] = audio
        audio_latent.pop("noise_mask", None)

        return (video_latent, audio_latent)


class MinimaxH3CombineAVLatent:
    """Recombine separate video/audio LATENTs into a joint MiniMax H3 AV LATENT."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT",),
                "audio_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "combine"
    CATEGORY = "video/MinimaxH3"
    DESCRIPTION = (
        "Recombines a plain 24-channel video LATENT and a plain 32-channel "
        "audio LATENT into a joint MiniMax H3 AV LATENT (NestedTensor[video, "
        "audio]) for AV-aware nodes such as the H3 AV VAE decode or refine pass."
    )

    def combine(self, video_latent, audio_latent):
        video = _validate_video(_plain_latent_samples(video_latent, label="video_latent"))
        audio = _validate_audio(_plain_latent_samples(audio_latent, label="audio_latent"))
        return ({"samples": _wrap_h3_samples(video, audio)},)


NODE_CLASS_MAPPINGS = {
    "MinimaxH3SplitAVLatent": MinimaxH3SplitAVLatent,
    "MinimaxH3CombineAVLatent": MinimaxH3CombineAVLatent,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxH3SplitAVLatent": "MiniMax H3 Split AV Latent",
    "MinimaxH3CombineAVLatent": "MiniMax H3 Combine AV Latent",
}
