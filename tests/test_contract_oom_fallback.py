"""Out-of-memory contract: a clip that doesn't fit still upscales, with a warning.

The fault is injected at the model boundary: LatentResizer3D refuses inputs as
long as the full clip, as a GPU without room for the full-clip pass would.
"""
from __future__ import annotations

import logging

import pytest
import torch

FULL_T = 40


@pytest.fixture
def full_clip_does_not_fit(lbh, monkeypatch):
    import comfy.model_management as mm

    real_forward = lbh.LatentResizer3D.forward

    def forward(self, x, *args, **kwargs):
        if x.shape[2] >= FULL_T:
            raise mm.OOM_EXCEPTION("injected: full clip does not fit")
        return real_forward(self, x, *args, **kwargs)

    monkeypatch.setattr(lbh.LatentResizer3D, "forward", forward)


def test_full_clip_oom_falls_back_and_still_returns_the_contract_output(
    lbh, tiny_checkpoint, full_clip_does_not_fit, caplog
):
    video = torch.randn(1, 24, FULL_T, 4, 6)

    with caplog.at_level(logging.WARNING):
        out = lbh.upscale_clean_video_exact(
            video, model_name=tiny_checkpoint, target_h=8, target_w=12, device="cpu", precision="fp32",
        )

    assert out.shape == (1, 24, FULL_T, 8, 12)
    assert out.dtype == video.dtype
    assert torch.isfinite(out).all()
    assert "temporal chunks" in caplog.text


def test_non_oom_model_errors_are_not_swallowed(lbh, tiny_checkpoint, monkeypatch):
    def broken(self, x, *args, **kwargs):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(lbh.LatentResizer3D, "forward", broken)

    with pytest.raises(RuntimeError, match="shape mismatch"):
        lbh.upscale_clean_video_exact(
            torch.randn(1, 24, 2, 4, 4), model_name=tiny_checkpoint, target_h=8, target_w=8,
            device="cpu", precision="fp32",
        )


@pytest.mark.gpu
def test_real_checkpoint_oom_fallback_on_cuda(lbh, real_checkpoint, full_clip_does_not_fit, caplog):
    video = torch.randn(1, 24, FULL_T, 12, 20)

    with caplog.at_level(logging.WARNING):
        out = lbh.upscale_clean_video_exact(
            video, model_name=real_checkpoint, target_h=24, target_w=40, device="cuda", precision="fp16",
        )

    assert out.shape == (1, 24, FULL_T, 24, 40)
    assert torch.isfinite(out).all()
    assert "temporal chunks" in caplog.text
