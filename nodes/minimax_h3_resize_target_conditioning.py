"""Resize target-grid MiniMax H3 conditioning to match an upscaled video latent.

Mirrors the conditioning-geometry step the integrated ``MiniMax H3 Latent
Upscaler + Refine (3D)`` node performs internally, for graphs that instead
split the AV latent (see ``minimax_h3_av_latent_split.py``), run the
standalone learned upscaler, and drive their own second-pass sampler.

``minimax_keyframes`` (FL2VA/I2VA/L2VA endpoint frames) are target-grid
conditions that must track the video latent currently being denoised, so
they are resized here. ``minimax_refs`` (REF2VA) intentionally keep their
own independent latent/RoPE geometry and are left untouched -- this node is
a no-op on pure REF2VA conditioning.
"""
from __future__ import annotations

from .minimax_h3_refine_support import (
    _h3_padded_spatial_size,
    _is_nested_tensor,
    _validate_h3_av_samples,
    _validate_video,
    resize_h3_target_conditioning,
)


def _extract_video(latent: dict):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("latent must be a LATENT dictionary containing 'samples'")
    samples = latent["samples"]
    if _is_nested_tensor(samples):
        video, _audio = _validate_h3_av_samples(samples)
        return video
    return _validate_video(samples)


class MinimaxH3ResizeTargetConditioning:
    """Resize ``minimax_keyframes`` inside CONDITIONING to match a LATENT's target grid."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "resize"
    CATEGORY = "video/MinimaxH3"
    DESCRIPTION = (
        "Resizes any MiniMax H3 'minimax_keyframes' (FL2VA/I2VA/L2VA endpoint-frame) "
        "latents inside CONDITIONING to match the target H/W of the given LATENT "
        "(e.g. after the standalone learned latent upscaler). 'minimax_refs' "
        "(REF2VA) are left unchanged since they keep their own geometry, so this "
        "node is safe to insert unconditionally before a second-pass sampler "
        "regardless of which mode produced the conditioning."
    )

    def resize(self, conditioning, latent):
        video = _extract_video(latent)
        target_h, target_w = _h3_padded_spatial_size(video)
        return (resize_h3_target_conditioning(conditioning, target_h, target_w),)


NODE_CLASS_MAPPINGS = {
    "MinimaxH3ResizeTargetConditioning": MinimaxH3ResizeTargetConditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxH3ResizeTargetConditioning": "MiniMax H3 Resize Target Conditioning",
}
