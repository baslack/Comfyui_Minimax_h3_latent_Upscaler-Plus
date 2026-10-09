"""Input contract: the video-only 3D upscaler rejects a joint H3 AV latent with a clear error."""
from __future__ import annotations

import pytest
import torch


def test_3d_upscaler_rejects_a_joint_av_latent_and_names_the_av_node(h3up):
    import comfy.nested_tensor

    node = h3up.NODE_CLASS_MAPPINGS["MinimaxH3LatentUpscaler3D"]
    joint = comfy.nested_tensor.NestedTensor([torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 8)])

    with pytest.raises(TypeError, match=r"Latent Upscaler \+ Refine \(3D\)"):
        node.execute(
            latent={"samples": joint},
            model_name="unused.safetensors",
            mode={"mode": "scale by multiplier", "scale": 2.0},
            align=16,
            keep_proportion=True,
            device="cpu",
            precision="fp32",
            offload_after_upscale=False,
        )
