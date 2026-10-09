"""Independent references and schema readers for the contract tests.

Nothing here reaches into the package's loader or caches: references are built
straight from checkpoint files and schemas are read through INPUT_TYPES(), the
same view ComfyUI's frontend uses to lay out widgets.
"""
from __future__ import annotations

import math

import torch

DYNAMIC_COMBO = "COMFY_DYNAMICCOMBO_V3"
_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", DYNAMIC_COMBO}


def load_reference_model(lbh, model_name, device, dtype):
    """Build LatentResizer3D directly from a checkpoint file."""
    from safetensors.torch import load_file
    import folder_paths

    state = load_file(folder_paths.get_full_path_or_raise("latent_upscale_models", model_name))
    if any(key.startswith("upscaler.") for key in state):
        state = {key[len("upscaler."):]: value for key, value in state.items() if key.startswith("upscaler.")}
    blocks = {
        side: len({key.split(".")[1] for key in state if key.startswith(f"{side}_blocks.") and ".in_layers." in key})
        for side in ("in", "out")
    }
    model = lbh.LatentResizer3D(
        in_channels=state["conv_in.weight"].shape[1],
        in_blocks=blocks["in"],
        out_blocks=blocks["out"],
        channels=state["conv_in.weight"].shape[0],
        dropout=0.0,
        temporal_every=2,
        temporal_kernel=next(v.shape[2] for k, v in state.items() if k.endswith("dwconv.weight")),
    )
    model.load_state_dict(state)
    return model.to(device=device, dtype=dtype).eval()


def reference_upscale(lbh, model, video, target_h, target_w, scale):
    """The documented math: normalize, run the model once over the full time axis, de-normalize."""
    mean = torch.tensor(lbh.LATENTS_MEAN, dtype=video.dtype, device=video.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(lbh.LATENTS_STD, dtype=video.dtype, device=video.device).view(1, -1, 1, 1, 1)
    with torch.inference_mode():
        out = model((video - mean) / std, scale=scale, target_size=(video.shape[2], target_h, target_w))
    return out * std + mean


def _is_widget(spec) -> bool:
    kind = spec[0]
    options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if options.get("forceInput"):
        return False
    return isinstance(kind, list) or kind in _WIDGET_TYPES


def map_saved_widgets(node_cls, saved_values):
    """Pair a saved workflow's positional widgets_values with the node's current widget inputs.

    Dynamic-combo children follow their combo, keyed by the saved combo value,
    which is how ComfyUI serializes them.
    """
    schema = node_cls.INPUT_TYPES()
    mapping = []

    def walk(section):
        for name, spec in section.items():
            if not _is_widget(spec):
                continue
            if len(mapping) >= len(saved_values):
                raise AssertionError(f"node now has more widgets than the saved workflow (next: {name})")
            value = saved_values[len(mapping)]
            mapping.append((name, spec, value))
            if spec[0] == DYNAMIC_COMBO:
                option = next((o for o in spec[1]["options"] if o["key"] == value), None)
                if option is None:
                    raise AssertionError(f"{name}: saved option {value!r} no longer exists")
                for key in ("required", "optional"):
                    walk(option["inputs"].get(key, {}))

    for key in ("required", "optional"):
        walk(schema.get(key, {}))
    return mapping


def assert_saved_value_fits(name, spec, value, *, skip_choices=("model_name",)):
    kind = spec[0]
    options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if kind == DYNAMIC_COMBO:
        assert value in [o["key"] for o in options["options"]], name
    elif isinstance(kind, list) or kind == "COMBO":
        if name not in skip_choices:
            choices = kind if isinstance(kind, list) else options["options"]
            assert value in choices, f"{name}: {value!r} not in {choices!r}"
    elif kind == "BOOLEAN":
        assert isinstance(value, bool), name
    elif kind in ("INT", "FLOAT"):
        assert isinstance(value, (int, float)) and not isinstance(value, bool), name
        assert options.get("min", -math.inf) <= value <= options.get("max", math.inf), name
    elif kind == "STRING":
        assert isinstance(value, str), name


def input_names(node_cls):
    schema = node_cls.INPUT_TYPES()
    return {key: list(schema.get(key, {})) for key in ("required", "optional")}
