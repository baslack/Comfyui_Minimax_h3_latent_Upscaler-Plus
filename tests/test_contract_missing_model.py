"""Missing-checkpoint contract: the error says where to get the model and where to put it."""
from __future__ import annotations

import pytest
import torch

SOURCE = "huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler"
NO_MODELS = "(place models in: models/latent_upscale_models)"


def test_3d_node_without_checkpoints_says_where_to_get_one(h3up):
    node = h3up.NODE_CLASS_MAPPINGS["MinimaxH3LatentUpscaler3D"]

    with pytest.raises(ValueError) as error:
        node.execute(
            latent={"samples": torch.randn(1, 24, 2, 4, 4)}, model_name=NO_MODELS,
            mode={"mode": "scale by multiplier", "scale": 2.0}, align=16, keep_proportion=True,
            device="cpu", precision="fp32",
        )

    assert SOURCE in str(error.value)
    assert "latent_upscale_models" in str(error.value)


def test_2d_node_without_checkpoints_says_where_to_get_one(h3up):
    node = h3up.NODE_CLASS_MAPPINGS["MinimaxH3LatentUpscalerNode2D"]()

    with pytest.raises(ValueError) as error:
        getattr(node, node.FUNCTION)(
            latent={"samples": torch.randn(1, 24, 4, 4)}, model_name=NO_MODELS, scale=2.0,
            device="cpu", precision="fp32",
        )

    assert SOURCE in str(error.value)
    assert "latent_upscale_models" in str(error.value)
