from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

import uvicorn

from inference_index_tts.server import (
    CORS_ORIGINS_ENV,
    ModelConfig,
    ModelVersion,
    ServerConfig,
    create_app,
    parse_cors_origins,
    resolve_model_dir,
)

SUPPORTED_MODELS: dict[str, ModelVersion] = {
    "index-tts-2": "2",
    "index-tts-2.5": "2.5",
}

REQUIRED_MODEL_FILES: dict[ModelVersion, tuple[str, ...]] = {
    "2": ("config.yaml", "bpe.model", "gpt.pth", "s2mel.pth", "wav2vec2bert_stats.pt"),
    "2.5": (
        "config.yaml",
        "codec.pth",
        "gpt.pth",
        "s2mel.pth",
        "multilingual_zh_ja_yue_char_del.tiktoken",
        "wav2vec2bert_stats.pt",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference-index-tts",
        description="Serve one or more IndexTTS models through an OpenAI-compatible endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME@file://PATH",
        help="Register index-tts-2 or index-tts-2.5; repeat to load both",
    )
    parser.add_argument(
        "--voice",
        action="append",
        default=[],
        metavar="NAME@file://PATH",
        help="Register a named reference audio; repeat for multiple voices",
    )
    parser.add_argument(
        "--default-language",
        default="auto",
        choices=("auto", "zh", "en", "ja", "es", "ar", "ko", "ru"),
        help="Language used when request extra.language is auto",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="Torch device, for example cuda:0, mps, or cpu")
    parser.add_argument(
        "--half",
        "--bf16",
        dest="half_precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use FP16 for v2 and BF16 for v2.5",
    )
    parser.add_argument("--cuda-kernel", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--deepspeed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--accel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--qwen-emo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load QwenEmotion so instructions and emotion mode=text are available",
    )
    parser.add_argument(
        "--max-reference-audio-bytes",
        type=int,
        default=20 * 1024 * 1024,
        help="Maximum decoded size of inline reference audio",
    )
    parser.add_argument(
        "--reference-cache-size",
        type=int,
        default=32,
        help="Number of decoded inline reference audio files kept in memory-local storage",
    )
    parser.add_argument("--api-key", default=None, help="Optional bearer token required from clients")
    parser.add_argument("--log-level", default="info")
    return parser


def parse_file_resource(value: str, *, resource: str) -> tuple[str, Path]:
    name, separator, uri = value.partition("@")
    if not separator or not name.strip() or not uri.strip():
        raise ValueError(f"{resource} must use NAME@file://PATH syntax: {value}")
    parsed = urlsplit(uri)
    if parsed.scheme != "file" or parsed.query or parsed.fragment:
        raise ValueError(f"{resource} path must be a file URI: {uri}")
    if parsed.netloc and parsed.path:
        raw_path = f"{parsed.netloc}{parsed.path}"
    elif parsed.netloc:
        raw_path = parsed.netloc
    else:
        raw_path = parsed.path
    raw_path = unquote(raw_path)
    if not raw_path:
        raise ValueError(f"{resource} file URI has no path: {uri}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return name.strip(), path.resolve()


def parse_models(values: list[str]) -> dict[str, ModelConfig]:
    models = {}
    for value in values:
        name, path = parse_file_resource(value, resource="model")
        version = SUPPORTED_MODELS.get(name)
        if version is None:
            supported = ", ".join(SUPPORTED_MODELS)
            raise ValueError(f"unsupported model '{name}'; supported models: {supported}")
        if name in models:
            raise ValueError(f"duplicate model name: {name}")
        model_dir = resolve_model_dir(path)
        missing = [filename for filename in REQUIRED_MODEL_FILES[version] if not (model_dir / filename).is_file()]
        if missing:
            raise ValueError(f"model '{name}' is incomplete; missing: {', '.join(missing)}")
        models[name] = ModelConfig(id=name, version=version, model_dir=model_dir)
    return models


def parse_voices(values: list[str]) -> dict[str, Path]:
    voices = {}
    for value in values:
        name, path = parse_file_resource(value, resource="voice")
        if not path.is_file():
            raise ValueError(f"voice reference audio does not exist: {path}")
        if name in voices:
            raise ValueError(f"duplicate voice name: {name}")
        voices[name] = path
    return voices


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_reference_audio_bytes <= 0:
        parser.error("--max-reference-audio-bytes must be greater than zero")
    if args.reference_cache_size <= 0:
        parser.error("--reference-cache-size must be greater than zero")
    try:
        models = parse_models(args.model)
        voices = parse_voices(args.voice)
    except ValueError as exc:
        parser.error(str(exc))
    config = ServerConfig(
        models=models,
        voices=voices,
        default_language=args.default_language,
        device=args.device,
        use_bf16=args.half_precision,
        use_cuda_kernel=args.cuda_kernel,
        use_deepspeed=args.deepspeed,
        use_accel=args.accel,
        use_torch_compile=args.torch_compile,
        use_qwen_emo=args.qwen_emo,
        max_reference_audio_bytes=args.max_reference_audio_bytes,
        reference_cache_size=args.reference_cache_size,
        api_key=args.api_key,
        cors_origins=parse_cors_origins(os.environ.get(CORS_ORIGINS_ENV)),
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
