# Tests

Tests run locally against a ComfyUI checkout, using ComfyUI's own Python
environment. `pytest` is the only extra package.

```powershell
cd <this repo>
$env:COMFYUI_PATH = "<ComfyUI checkout>"          # e.g. D:\ComfyUI
$env:PYTHONPATH   = "$env:COMFYUI_PATH;$PWD"
$env:CUDA_VISIBLE_DEVICES = "-1"                  # hide the GPU (not "": that deletes the variable)
<ComfyUI python> -m pytest tests -m "not gpu"     # fast, CPU only
```

To also run the `gpu` tests, `Remove-Item Env:CUDA_VISIBLE_DEVICES` and run
`<ComfyUI python> -m pytest tests`, and only while ComfyUI isn't generating.
Even CPU-only runs create a CUDA context unless the GPU is hidden.

Run from the repo root and pass `tests` as shown: `tests/pytest.ini` then
becomes the config (not ComfyUI's), and `test_native_comfyui_fixture.py`
expects `import nodes` to resolve to this package.

- `test_contract_*.py` test the contract (node schemas and saved-workflow
  widget layout, provider and upscale outputs), not internals. CPU tests use a
  tiny random checkpoint written to a temp folder.
- `gpu` tests need CUDA and a MiniMax H3 latent upscaler checkpoint
  (`minimax_h3_latent_upscaler_3d*.safetensors`) in `models/latent_upscale_models`
  (subfolders are fine), or set `H3_UPSCALER_TEST_MODEL` to its listed name.
- Tests may not open network connections; any non-loopback connect fails.
