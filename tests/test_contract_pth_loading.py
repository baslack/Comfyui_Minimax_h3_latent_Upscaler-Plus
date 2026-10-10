"""Checkpoint-format contract: .pth files load as plain tensors only, never arbitrary pickles."""
from __future__ import annotations

import shutil

import pytest
import torch
from safetensors.torch import load_file


class NotATensor:
    """Anything a pickle could smuggle in; loading must refuse it instead of constructing it."""


@pytest.fixture
def pth_root(tmp_path, monkeypatch):
    import folder_paths

    paths, extensions = folder_paths.folder_names_and_paths["latent_upscale_models"]
    monkeypatch.setitem(
        folder_paths.folder_names_and_paths, "latent_upscale_models", ([str(tmp_path), *paths], extensions)
    )
    return tmp_path


def test_plain_tensor_pth_loads_like_the_safetensors_checkpoint(lbh, tiny_checkpoint, _tiny_checkpoint_dir, pth_root):
    shutil.copy(_tiny_checkpoint_dir / tiny_checkpoint, pth_root / tiny_checkpoint)
    torch.save(load_file(str(_tiny_checkpoint_dir / tiny_checkpoint)), pth_root / "tiny_plain.pth")
    video = torch.randn(1, 24, 3, 4, 4)
    kwargs = dict(target_h=8, target_w=8, device="cpu", precision="fp32")

    from_pth = lbh.upscale_clean_video_exact(video, model_name="tiny_plain.pth", **kwargs)
    from_safetensors = lbh.upscale_clean_video_exact(video, model_name=tiny_checkpoint, **kwargs)

    # Same weights, separately loaded: CPU kernels may differ in the last bits.
    assert torch.allclose(from_pth, from_safetensors, atol=1e-5, rtol=1e-5)


def test_pth_carrying_a_pickled_object_is_refused(lbh, tiny_checkpoint, _tiny_checkpoint_dir, pth_root):
    state = load_file(str(_tiny_checkpoint_dir / tiny_checkpoint))
    torch.save({**state, "payload": NotATensor()}, pth_root / "tiny_pickled.pth")

    with pytest.raises(Exception, match="(?i)weights.only"):
        lbh.upscale_clean_video_exact(
            torch.randn(1, 24, 2, 4, 4), model_name="tiny_pickled.pth", target_h=8, target_w=8,
            device="cpu", precision="fp32",
        )
