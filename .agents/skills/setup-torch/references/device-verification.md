# Device verification

Run every command from the repository root through the same selected Pixi manifest and environment that owns Torch. The unqualified commands below assume Pixi can automatically discover the selected local manifest from the repository root. For a manifest outside that location or a repository with multiple manifests, add `--manifest-path MANIFEST_PATH` immediately after the `run` or `lock` subcommand. For a non-default environment, add `--environment ENVIRONMENT` to each `pixi run`. Use real names without shell angle-bracket placeholders.

## Version and backend report

```bash
pixi run python -c "import torch, torchvision, torchaudio; mps = getattr(torch.backends, 'mps', None); is_built = getattr(mps, 'is_built', None); is_available = getattr(mps, 'is_available', None); print({'torch': torch.__version__, 'torchvision': torchvision.__version__, 'torchaudio': torchaudio.__version__, 'torch_cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'cuda_devices': torch.cuda.device_count(), 'mps_built': bool(callable(is_built) and is_built()), 'mps_available': bool(callable(is_available) and is_available())})"
```

`torch.version.cuda` identifies the CUDA runtime family used to build the installed Torch distribution, or is `None` for a non-CUDA build. It is distinct from both the latest/maximum driver-supported level reported as `CUDA UMD Version` or legacy `CUDA Version` by `nvidia-smi` and a local toolkit reported by `nvcc --version`.

## Compute smoke test

Create a temporary `verify_torch_device.py` with this exact code, then run `pixi run python verify_torch_device.py` and remove the temporary file after recording the result:

```python
import torch


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    is_built = getattr(backend, "is_built", None)
    is_available = getattr(backend, "is_available", None)
    return bool(callable(is_built) and is_built() and callable(is_available) and is_available())


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


device = select_device()
x = torch.tensor([1.0, 2.0], device=device)
y = x + 1

if device.type == "cuda":
    torch.cuda.synchronize(device)
elif device.type == "mps":
    mps_runtime = getattr(torch, "mps", None)
    synchronize = getattr(mps_runtime, "synchronize", None)
    if callable(synchronize):
        synchronize()

print({"actual_device": str(device), "result": y.detach().cpu().tolist()})
```

The result must be `[2.0, 3.0]`. The printed `actual_device` is the verified runtime choice. If it differs from the candidate backend, report the difference rather than silently accepting a fallback.

## Lock and project checks

With a local manifest and the default environment:

```bash
pixi lock --check
pixi run --environment default --locked test
```

For another manifest or environment, use its real path and name consistently, for example `pixi lock --manifest-path MANIFEST_PATH --check` and `pixi run --manifest-path MANIFEST_PATH --environment ENVIRONMENT --locked test`. Run device checks only on a host compatible with the selected Pixi platform; solving or installing a foreign target is not proof that its runtime works.

## Diagnosis order

If the candidate accelerator is unavailable, capture and compare:

1. detector JSON, including raw command status and `candidate_reason`
2. `nvidia-smi` summary and GPU-query output; when compilation is relevant, capture `nvcc --version` separately
3. installed package versions, `torch.version.cuda`, and the exact wheel channel
4. `torch.cuda.is_available()` and device count, or MPS built/available flags
5. selected manifest, environment, Python/ABI, and target platform
6. container, VM, or WSL device passthrough and host-driver state
7. filenames proving a complete trio exists for the selected release group

Official references:

- [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
