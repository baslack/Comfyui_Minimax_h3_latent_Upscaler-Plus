"""Model lookup contract: checkpoints anywhere under a registered latent_upscale_models root."""
from __future__ import annotations

import os
import shutil

import torch


def test_checkpoint_in_a_subfolder_of_any_root_is_listed_and_loads(
    h3up, lbh, tiny_checkpoint, _tiny_checkpoint_dir, tmp_path, monkeypatch
):
    import folder_paths

    (tmp_path / "_h3").mkdir()
    shutil.copy(_tiny_checkpoint_dir / tiny_checkpoint, tmp_path / "_h3" / tiny_checkpoint)
    paths, extensions = folder_paths.folder_names_and_paths["latent_upscale_models"]
    monkeypatch.setitem(
        folder_paths.folder_names_and_paths,
        "latent_upscale_models",
        ([*paths, str(tmp_path)], extensions),
    )
    name = os.path.join("_h3", tiny_checkpoint)

    for key in ("MinimaxH3LatentUpscaler3D", "MinimaxH3LatentUpscaler3DProvider"):
        spec = h3up.NODE_CLASS_MAPPINGS[key].INPUT_TYPES()["required"]["model_name"]
        options = spec[0] if isinstance(spec[0], list) else spec[1]["options"]
        assert name in options, key

    out = lbh.upscale_clean_video_exact(
        torch.randn(1, 24, 2, 4, 4), model_name=name, target_h=8, target_w=8,
        device="cpu", precision="fp32",
    )
    assert out.shape == (1, 24, 2, 8, 8)
