"""Upscale contract: outputs of the public upscale entry points.

CPU tests run a tiny random checkpoint through the real code path; gpu tests
use the real MiniMax H3 latent upscaler from latent_upscale_models.
"""
from __future__ import annotations

import pytest
import torch

from contract_helpers import load_reference_model, reference_upscale


def _video(*shape, seed=7, dtype=torch.float32):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def test_exact_upscale_is_the_full_clip_model_math(lbh, tiny_checkpoint):
    video = _video(1, 24, 7, 4, 6)
    before = video.clone()

    out = lbh.upscale_clean_video_exact(
        video, model_name=tiny_checkpoint, target_h=8, target_w=12, device="cpu", precision="fp32",
    )

    model = load_reference_model(lbh, tiny_checkpoint, "cpu", torch.float32)
    expected = reference_upscale(lbh, model, video, 8, 12, scale=2.0)
    assert out.shape == (1, 24, 7, 8, 12)
    assert out.dtype == video.dtype and out.device == video.device
    # Independently built model: identical math, possibly different CPU kernels.
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)
    assert torch.equal(video, before)


def test_exact_upscale_preserves_input_dtype(lbh, tiny_checkpoint):
    video = _video(1, 24, 2, 4, 4).to(torch.bfloat16)

    out = lbh.upscale_clean_video_exact(
        video, model_name=tiny_checkpoint, target_h=6, target_w=6, device="cpu", precision="fp32",
    )

    assert out.dtype == torch.bfloat16
    assert out.shape == (1, 24, 2, 6, 6)


def test_offload_option_does_not_change_output(lbh, tiny_checkpoint):
    video = _video(1, 24, 7, 4, 4)
    kwargs = dict(model_name=tiny_checkpoint, target_h=8, target_w=8, device="cpu", precision="fp32")

    kept = lbh.upscale_clean_video_exact(video, offload_after_upscale=False, **kwargs)
    offloaded = lbh.upscale_clean_video_exact(video, offload_after_upscale=True, **kwargs)
    again = lbh.upscale_clean_video_exact(video, offload_after_upscale=False, **kwargs)

    assert torch.equal(kept, offloaded)
    assert torch.equal(kept, again)


def test_provider_output_equals_exact_api(lbh, provider_module, tiny_checkpoint):
    video = _video(2, 24, 3, 4, 6)
    provider = provider_module.H3LatentUpscalerProvider(
        model_name=tiny_checkpoint, device="cpu", precision="fp32",
    )

    out = provider.upscale_clean_video(video, target_h=8, target_w=10)

    expected = lbh.upscale_clean_video_exact(
        video, model_name=tiny_checkpoint, target_h=8, target_w=10, device="cpu", precision="fp32",
    )
    assert torch.equal(out, expected)


def test_3d_node_upscales_on_the_aligned_latent_grid(h3up, lbh, tiny_checkpoint):
    node = h3up.NODE_CLASS_MAPPINGS["MinimaxH3LatentUpscaler3D"]
    video = _video(1, 24, 7, 4, 6)

    result = node.execute(
        latent={"samples": video},
        model_name=tiny_checkpoint,
        mode={"mode": "scale by multiplier", "scale": 2.0},
        align=16,
        keep_proportion=True,
        device="cpu",
        precision="fp32",
        offload_after_upscale=False,
    )
    out = result.result[0]["samples"]

    expected = lbh.upscale_clean_video_exact(
        video, model_name=tiny_checkpoint, target_h=8, target_w=12, device="cpu",
        precision="fp32", scale_embedding=2.0,
    )
    assert out.shape == (1, 24, 7, 8, 12)
    assert torch.equal(out, expected)


@pytest.mark.gpu
def test_real_checkpoint_on_cuda_is_the_full_clip_model_math(lbh, real_checkpoint):
    video = _video(1, 24, 12, 24, 40)

    out = lbh.upscale_clean_video_exact(
        video, model_name=real_checkpoint, target_h=48, target_w=80, device="cuda", precision="fp16",
    )

    model = load_reference_model(lbh, real_checkpoint, "cuda", torch.float16)
    expected = reference_upscale(lbh, model, video.to("cuda", torch.float16), 48, 80, scale=2.0).float().cpu()
    del model
    assert out.shape == (1, 24, 12, 48, 80)
    assert out.dtype == torch.float32 and out.device.type == "cpu"
    assert torch.allclose(out, expected, atol=2e-2, rtol=1e-2)


@pytest.mark.gpu
def test_offload_after_upscale_releases_the_upscaler_vram(lbh, real_checkpoint):
    video = _video(1, 24, 2, 8, 8)
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()

    lbh.upscale_clean_video_exact(
        video, model_name=real_checkpoint, target_h=16, target_w=16, device="cuda",
        precision="fp16", offload_after_upscale=True,
    )

    assert torch.cuda.memory_allocated() - baseline < 64 * 1024 * 1024
