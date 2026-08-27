# Visual Studio Code setup

Install the extensions recommended by this workspace when Visual Studio Code prompts you. You can also open **Extensions: Show Recommended Extensions** from the Command Palette.

Create the locked development environment from the repository root:

```shell
pixi install --locked
```

For editor language services and debugging, run **Python: Select Interpreter** and choose the interpreter inside Pixi's default environment:

- POSIX: `.pixi/envs/default/bin/python`
- Windows: `.pixi\envs\default\python.exe`

Do not depend on that manually selected interpreter for project commands. Terminal commands, editor tasks, tests, and CI should run through `pixi run`, for example:

```shell
pixi run --locked check
pixi run --locked test
```

## Configuration ownership

- `pyproject.toml` owns the Pixi environment, development tools, tasks, package metadata, and build configuration. `pixi.lock` records the exact resolved environment.
- `ruff.toml` owns formatting, import organization, and linting.
- `ty.toml` owns static type checking.
- `.vscode/` owns workspace extension recommendations and editor behavior.

Keeping commands behind Pixi tasks gives local shells and Visual Studio Code workflows the same tool versions and behavior.
