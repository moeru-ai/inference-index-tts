# PyTorch compatibility sources

Use sources in this order:

1. [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/) for release-group compatibility and official installation commands.
2. [PyTorch wheel root](https://download.pytorch.org/whl/) and the selected channel's package indexes for actual wheel filenames.
3. [Pixi manifest reference](https://pixi.sh/latest/reference/pixi_manifest/#pypi-dependencies) for per-package `index`, target tables, feature ownership, and environment definitions.
4. [Pixi multi-environment tutorial](https://pixi.sh/latest/tutorials/multi_environment/) for default-feature and `no-default-feature` behavior.
5. [Pixi lock CLI](https://pixi.sh/latest/reference/cli/pixi/lock/) and [Pixi run CLI](https://pixi.sh/latest/reference/cli/pixi/run/) for manifest, environment, platform, and lock-state flags.

The version page determines which `torch`, `torchvision`, and `torchaudio` releases belong together. Index listings only prove that a listed release has a wheel matching the requested Python ABI, platform, and architecture; they do not create a compatibility relationship.

In Pixi, root target dependency tables belong to the default feature, while named-feature target tables affect only environments containing that feature. `pixi lock` resolves the workspace, and `pixi run --locked` selects a runtime environment without allowing a stale lock file to be rewritten.

For an index base `https://download.pytorch.org/whl/cu130`, inspect:

```text
https://download.pytorch.org/whl/cu130/torch/
https://download.pytorch.org/whl/cu130/torchvision/
https://download.pytorch.org/whl/cu130/torchaudio/
```

Wheel checks must match all of:

- official release-group version
- Python tag such as `cp313`
- operating-system tag
- architecture tag
- selected CPU or CUDA local-version build

`cu132` is an index name, not proof of a complete trio. If a current Torch release appears there without matching torchvision and torchaudio artifacts, return an incompatibility result.
