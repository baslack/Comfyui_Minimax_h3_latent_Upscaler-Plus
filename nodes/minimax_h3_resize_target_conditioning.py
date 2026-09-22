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

Without a ``learned_upscaler`` provider, resizing falls back to the same
nearest-neighbor interpolation the integrated refine node uses on target
keyframes. That reproduces MiniMax H3's own conditioning geometry exactly
but does not reconstruct real high-frequency detail: a 24-channel latent
cell is not a pixel, and duplicating it into a 2x2 (or larger) block does
not encode the higher-resolution structure a learned upscale would. Wire in
a ``MinimaxH3LatentUpscaler3DProvider`` (the same learned checkpoint used
for the video) to upscale keyframe latents with it instead, matching the
quality of the surrounding video and avoiding a visibly degraded anchor
frame.
"""
from __future__ import annotations

import torch

from .minimax_h3_refine_support import (
    H3_VIDEO_CHANNELS,
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


def _resize_target_keyframe_learned(block: dict, target_h: int, target_w: int, provider) -> dict:
    """Resize one target-grid keyframe latent with the learned upscaler, not nearest."""
    out = dict(block)
    latent = out.get("latent")
    if latent is None:
        return out
    if not isinstance(latent, torch.Tensor):
        raise TypeError(
            f"MiniMax keyframe latent must be torch.Tensor, got {type(latent).__name__}"
        )
    if latent.ndim not in (4, 5) or latent.shape[1] != H3_VIDEO_CHANNELS:
        raise ValueError(
            "MiniMax H3 keyframe visual latent must be Bx24xHxW or Bx24xTxHxW, "
            f"got {tuple(latent.shape)}."
        )
    added_t = latent.ndim == 4
    video = latent.unsqueeze(2) if added_t else latent
    resized = provider.upscale_clean_video(video, target_h=target_h, target_w=target_w)
    if added_t:
        resized = resized.squeeze(2)
    out["latent"] = resized
    if "latent_h" in out:
        out["latent_h"] = int(target_h)
    if "latent_w" in out:
        out["latent_w"] = int(target_w)
    if "latent_t" in out and resized.ndim == 5:
        out["latent_t"] = int(resized.shape[2])
    return out


def _resize_h3_target_conditioning_learned(
    conditioning: list | None,
    target_h: int,
    target_w: int,
    provider,
) -> list | None:
    """Same contract as ``resize_h3_target_conditioning`` but using a learned upscale."""
    if conditioning is None:
        return None
    result = []
    for entry in conditioning:
        if (
            not isinstance(entry, (list, tuple))
            or len(entry) < 2
            or not isinstance(entry[1], dict)
        ):
            result.append(entry)
            continue
        metadata = entry[1]
        new_meta = metadata.copy()
        keyframes = metadata.get("minimax_keyframes")
        if keyframes is not None:
            new_meta["minimax_keyframes"] = [
                _resize_target_keyframe_learned(block, target_h, target_w, provider)
                for block in keyframes
            ]
        rebuilt = [entry[0], new_meta, *entry[2:]]
        result.append(tuple(rebuilt) if isinstance(entry, tuple) else rebuilt)
    return result


class MinimaxH3ResizeTargetConditioning:
    """Resize ``minimax_keyframes`` inside CONDITIONING to match a LATENT's target grid."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
            },
            "optional": {
                "learned_upscaler": ("H3_LATENT_UPSCALER",),
            },
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
        "regardless of which mode produced the conditioning. Connect a "
        "'learned_upscaler' (MinimaxH3LatentUpscaler3DProvider) to resize keyframes "
        "with the same learned network used for the video instead of nearest-neighbor "
        "interpolation -- nearest-neighbor duplicates latent cells rather than "
        "reconstructing detail and visibly degrades the anchored frame."
    )

    def resize(self, conditioning, latent, learned_upscaler=None):
        video = _extract_video(latent)
        target_h, target_w = _h3_padded_spatial_size(video)
        if learned_upscaler is not None:
            return (
                _resize_h3_target_conditioning_learned(
                    conditioning, target_h, target_w, learned_upscaler
                ),
            )
        return (resize_h3_target_conditioning(conditioning, target_h, target_w),)


NODE_CLASS_MAPPINGS = {
    "MinimaxH3ResizeTargetConditioning": MinimaxH3ResizeTargetConditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxH3ResizeTargetConditioning": "MiniMax H3 Resize Target Conditioning",
}
