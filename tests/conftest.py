from __future__ import annotations

import importlib.util
import ipaddress
import os
import socket
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

comfyui_path = os.environ.get("COMFYUI_PATH")
if comfyui_path and comfyui_path not in sys.path:
    sys.path.insert(0, comfyui_path)

# Native ComfyUI source-contract tests run on CPU-only GitHub runners. Prime
# ComfyUI's public CLI args once with --cpu before source imports can initialize
# model_management. Restore pytest's argv immediately afterward.
if comfyui_path and not torch.cuda.is_available():
    original_argv = sys.argv[:]
    try:
        sys.argv[:] = [original_argv[0], "--cpu"]
        import comfy.options

        comfy.options.enable_args_parsing()
        import comfy.cli_args
    finally:
        sys.argv[:] = original_argv

    # Keep the reviewed ComfyUI revision matrix deterministic while using the
    # same minimal dependency versions as Spectrum's native fixture harness.
    import comfy_kitchen

    if not hasattr(comfy_kitchen, "int8_attention_is_available"):
        comfy_kitchen.int8_attention_is_available = lambda: False


CONTRACT_PACKAGE = "h3_upscaler_plus_contract"
TINY_CHECKPOINT = "tiny_h3_latent_upscaler_3d.safetensors"
REAL_CHECKPOINT_STEM = "minimax_h3_latent_upscaler_3d"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: needs CUDA and a real MiniMax H3 latent upscaler checkpoint in latent_upscale_models",
    )


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="CUDA is not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _is_loopback(address) -> bool:
    host = address[0] if isinstance(address, tuple) else address
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Nothing in this package may reach the network; loopback stays usable for asyncio."""

    def connect(self, address):
        if not _is_loopback(address):
            raise RuntimeError(f"tests must not open network connections (attempted {address!r})")
        return _real_connect(self, address)

    def connect_ex(self, address):
        if not _is_loopback(address):
            raise RuntimeError(f"tests must not open network connections (attempted {address!r})")
        return _real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


@pytest.fixture(scope="session")
def h3up():
    """The custom node package, loaded the way ComfyUI loads custom_nodes/<pkg>/__init__.py."""
    pytest.importorskip("folder_paths", reason="set COMFYUI_PATH to a ComfyUI checkout")
    if CONTRACT_PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            CONTRACT_PACKAGE, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[CONTRACT_PACKAGE] = module
        spec.loader.exec_module(module)
    return sys.modules[CONTRACT_PACKAGE]


@pytest.fixture(scope="session")
def lbh(h3up):
    return importlib.import_module(f"{CONTRACT_PACKAGE}.nodes.minimax_h3_latent_upscaler_3d")


@pytest.fixture(scope="session")
def provider_module(h3up):
    return importlib.import_module(f"{CONTRACT_PACKAGE}.nodes.minimax_h3_handoff_provider")


@pytest.fixture(scope="session")
def _tiny_checkpoint_dir(lbh, tmp_path_factory):
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(1234)
    model = lbh.LatentResizer3D(
        in_channels=24, in_blocks=1, out_blocks=1, channels=32, dropout=0.0,
        temporal_every=2, temporal_kernel=5,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    folder = tmp_path_factory.mktemp("latent_upscale_models")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(folder / TINY_CHECKPOINT))
    return folder


@pytest.fixture
def tiny_checkpoint(_tiny_checkpoint_dir, monkeypatch):
    """A small random LatentResizer3D checkpoint, first in latent_upscale_models for one test.

    Nothing is written to the real models directory, and the real roots are
    restored after the test.
    """
    import folder_paths

    paths, extensions = folder_paths.folder_names_and_paths["latent_upscale_models"]
    monkeypatch.setitem(
        folder_paths.folder_names_and_paths,
        "latent_upscale_models",
        ([str(_tiny_checkpoint_dir), *paths], extensions),
    )
    return TINY_CHECKPOINT


@pytest.fixture(scope="session")
def real_checkpoint(h3up):
    """Name of the real MiniMax H3 latent upscaler as listed by ComfyUI (gpu tests only)."""
    import folder_paths

    name = os.environ.get("H3_UPSCALER_TEST_MODEL")
    if name:
        return name
    for candidate in folder_paths.get_filename_list("latent_upscale_models"):
        if Path(candidate.replace("\\", "/")).name.startswith(REAL_CHECKPOINT_STEM):
            return candidate
    pytest.skip("no MiniMax H3 latent upscaler checkpoint found in latent_upscale_models")
