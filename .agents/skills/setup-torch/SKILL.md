---
name: setup-torch
description: Use when a Pixi-managed project needs PyTorch but the host backend is not yet selected, including NVIDIA driver or GPU inspection, Apple Silicon MPS detection, CPU fallback, or post-install device verification.
---

# Set Up Torch

**REQUIRED SUB-SKILL:** Invoke and follow `$install-torch` for release-group and wheel-index validation and for every manifest change. `$install-torch` in turn requires `$use-pixi`.

Detect the host, select only a candidate backend, delegate installation, and prove the actual runtime device with computation.

## Inspect project context

1. Read the selected Pixi manifest and `pixi.lock`. Record the selected environment, target platform, Python interpreter family and major/minor constraint, features, and dependency ownership.
2. Confirm that the target platform is runnable on the current host. Cross-platform lock resolution is not runtime verification.
3. From the repository root, run the detector through the selected Pixi environment. For this template's default environment:

```bash
pixi run python .agents/skills/setup-torch/scripts/detect-environment.py --json
```

For another manifest, environment, or host platform, add the corresponding `--manifest-path`, `--environment`, and `--platform` selectors consistently to every Pixi command.

4. Optionally run these read-only probes directly:

```bash
nvidia-smi
nvtop --version
```

Treat a missing or failing command only as evidence about the current shell. It does not prove that NVIDIA hardware is absent. Check driver installation and container, VM, or WSL device passthrough before ruling out the device.

## Interpret detector output

Treat `candidate_backend`, `candidate_reason`, and `verification_required` as a proposal, not a working-device result.

- `cuda`: Require a successful `nvidia-smi` summary and non-empty GPU enumeration. Read current `KMD Version` and `CUDA UMD Version` labels as well as legacy `Driver Version` and `CUDA Version` labels.
- `mps`: Treat Darwin on arm64 or aarch64 as an Apple Silicon signal. Do not claim MPS availability before installing and importing Torch.
- `cpu`: Use as the safe fallback whenever no accelerator is confirmed. Explain any inconclusive NVIDIA probe rather than claiming that no GPU exists.

`CUDA UMD Version` or legacy `CUDA Version` is the latest/maximum CUDA level supported by the NVIDIA driver. It is not the locally installed CUDA toolkit, does not prove that `nvcc` exists, and does not prove that a Torch CUDA build can enumerate a device. Prebuilt Torch wheels normally provide their CUDA runtime dependencies; install a local CUDA toolkit only when the project separately compiles CUDA extensions or other CUDA code.

## Select and delegate

1. Read the current official PyTorch previous-versions page and NVIDIA compatibility documentation when CUDA is a candidate.
2. Select the newest published, complete `torch` / `torchvision` / `torchaudio` release group that supports the selected Python and platform.
3. For CUDA, select an exact official channel such as `cu130` that does not exceed the driver-supported maximum. Treat an alphanumeric, `N/A`, missing, or otherwise incomparable driver maximum as inconclusive; diagnose or ask rather than guessing.
4. For Apple Silicon, select the official macOS trio and use MPS only as the candidate. Otherwise select the official CPU trio.
5. Invoke `$install-torch` with all of these explicit values:

   - selected Pixi manifest and environment
   - target Pixi platform and Python major/minor
   - `cpu`, `mps`, or the exact CUDA channel
   - exact `torch`, `torchvision`, and `torchaudio` versions from one official release group

Require `$install-torch` to prove that all three matching wheels exist for the Python ABI, platform, and selected channel. If any member is missing, try an older complete official group or stop. Do not mix release rows, silently change Python, write Torch dependencies directly, or repeat `$install-torch`'s index-inspection logic here.

## Verify actual runtime behavior

After `$install-torch` succeeds, read [device verification](references/device-verification.md) and run its commands through the same manifest and environment.

1. Import all three packages and report their versions.
2. Report `torch.version.cuda`, CUDA availability and device count, and MPS built/available flags.
3. Select the actual device in this order: available CUDA, available MPS, then CPU.
4. Allocate tensors on that actual device, add them, synchronize CUDA or MPS when supported, and copy the result to CPU for comparison.
5. Compare the actual device with the candidate. If they differ, report the mismatch and relevant driver, passthrough, Python, platform, build, and wheel-channel evidence. Never describe an unverified accelerator as usable.
6. Run `pixi lock --check` with the selected manifest and run the project's tests through the selected environment.

Report the candidate, actual device, computation result, release trio, wheel source, manifest/environment/platform, lock state, and test result.
