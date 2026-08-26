---
name: install-torch
description: Use when an explicit CPU, MPS, or CUDA target has already been chosen and a Pixi workspace must install a mutually compatible torch, torchvision, and torchaudio trio after checking official versions and wheel availability. Do not use this Skill to detect GPUs or choose the target.
---

# Install Torch

REQUIRED BACKGROUND: Read and follow `$use-pixi` before changing a Pixi manifest.

Install all three PyTorch distributions only after the caller supplies the target platform and one of: CPU, MPS, or a concrete CUDA wheel channel such as `cu130`.

## Required input

Collect these values before editing:

- selected Pixi manifest and environment
- Pixi platform: `linux-64`, `win-64`, or `osx-arm64`
- interpreter family, Python major/minor version, and expected Python/ABI tags; the inspector supports standard GIL-enabled release CPython only, derives `cpNN` from major/minor, and accepts that Python tag with `cpNN`, `abi3`, or `none` ABI
- explicit backend: CPU, MPS, or a concrete CUDA channel

If the backend is not explicit, stop and invoke `$setup-torch`; do not inspect GPU hardware here.

This template uses standard GIL-enabled release CPython. Refuse automated inspector validation for PyPy, free-threaded tags such as `cp313t`, debug ABI tags such as `cp313d`, or other interpreter variants. Do not pass `3.13` and silently treat a variant as the standard `cp313` target.

## Resolve a compatible trio

Before choosing versions, read [PyTorch compatibility sources](references/pytorch-sources.md) and follow its official source ordering.

1. Read the current [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/) page. Treat one published row/command as the authoritative `torch` / `torchvision` / `torchaudio` release group.
2. Prefer the newest complete release group supporting the requested Python, platform, and backend. A newer `torch` release without a matching `torchaudio` release is not a complete trio.
3. Map CPU to `https://download.pytorch.org/whl/cpu`. For macOS/MPS, use the default PyPI simple index unless the official command says otherwise. Map CUDA channel `cuXYZ` to `https://download.pytorch.org/whl/cuXYZ`.
4. Inspect each package index directly. A reproducible first check is:

```bash
curl -fsSL https://download.pytorch.org/whl/cu130/torch/
curl -fsSL https://download.pytorch.org/whl/cu130/torchvision/
curl -fsSL https://download.pytorch.org/whl/cu130/torchaudio/
```

5. From the repository root, run the inspector with every required input. For example:

```bash
python .agents/skills/install-torch/scripts/inspect-wheel-index.py \
  --index-url https://download.pytorch.org/whl/cu130 \
  --python-version 3.13 \
  --platform linux-64 \
  --torch-version 2.11.0 \
  --torchvision-version 0.26.0 \
  --torchaudio-version 2.11.0 \
  --json
```

For macOS/MPS, pass `--index-url https://pypi.org/simple`. Exit 0 means a complete trio exists. Exit 1 means a wheel is missing or the candidates have no channel-consistent trio. Exit 2 means invalid input, a network or inspection failure, or multiple compatible local-version channels are ambiguous and require an explicit `+local` version. For exits 0 and 1, inspect the JSON `compatible`, `missing`, `matches`, `selected_channel`, and `channel_issue` fields; exit 2 reports the inspection error on stderr.
6. Stop on a missing package. Do not mix release rows, change Python silently, or omit torchaudio.

## Change the Pixi manifest

Pin the verified versions under the selected platform's `pypi-dependencies`. In `pixi.toml`, a CUDA target looks like:

```toml
[target.linux-64.pypi-dependencies]
torch = { version = "==2.11.0", index = "https://download.pytorch.org/whl/cu130" }
torchvision = { version = "==0.26.0", index = "https://download.pytorch.org/whl/cu130" }
torchaudio = { version = "==2.11.0", index = "https://download.pytorch.org/whl/cu130" }
```

The root `[target.<platform>.pypi-dependencies]` table belongs to the default feature. It affects the selected environment only when that environment includes the default feature.

If the selected environment sets `no-default-feature = true`, or Torch should remain optional, put the pins under a named feature already owned by that environment:

```toml
[feature.torch.target.linux-64.pypi-dependencies]
torch = { version = "==2.11.0", index = "https://download.pytorch.org/whl/cu130" }
torchvision = { version = "==0.26.0", index = "https://download.pytorch.org/whl/cu130" }
torchaudio = { version = "==2.11.0", index = "https://download.pytorch.org/whl/cu130" }
```

Here, `torch` is an example owning feature name that must already be present in the selected environment. Do not silently change environment membership: choose an owning feature already present in the environment, or ask before changing the environment definition. For a Pixi manifest embedded in `pyproject.toml`, prefix the whole table path with `tool.pixi`, producing `[tool.pixi.target.linux-64.pypi-dependencies]` for the default-feature example or `[tool.pixi.feature.torch.target.linux-64.pypi-dependencies]` for the named-feature example. Adapt the concrete platform and feature names to the selected environment. For macOS/MPS, omit a custom index when the official command uses PyPI. Change only the requested target; do not add a CUDA toolkit unless the project independently requires local CUDA compilation.

## Lock and verify

Run:

```bash
pixi lock --manifest-path pixi.toml
pixi lock --manifest-path pixi.toml --check
pixi run --manifest-path pixi.toml --environment default --platform linux-64 --locked python -c "import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__)"
```

Replace `pixi.toml`, `default`, and `linux-64` with the selected manifest, environment, and platform. `pixi lock` resolves all declared platforms and environments in the workspace. Run the import command only on a host or CI runner compatible with the selected platform; a non-host `pixi run --platform` invocation does not prove runtime usability. `--locked` prevents verification from silently rewriting a stale lock file.

Verify the installed versions belong to the selected release group. Report the manifest diff, owning feature and environment, index, Python/platform tags, lock result, and import versions. Do not report device availability here; `$setup-torch` owns device verification.
