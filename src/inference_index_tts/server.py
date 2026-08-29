from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import re
import shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Protocol

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
ReferenceAudioFormat = Literal["wav", "mp3", "flac", "ogg", "opus", "aac", "m4a"]
ModelVersion = Literal["2", "2.5"]
Language = Literal["auto", "zh", "en", "ja", "es", "ar", "ko", "ru"]

CORS_ORIGINS_ENV = "INFERENCE_INDEX_TTS_ORIGINS"
DEFAULT_CORS_ORIGINS = (
    *(
        pattern
        for host in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]")
        for pattern in (
            f"http://{host}",
            f"https://{host}",
            f"http://{host}:*",
            f"https://{host}:*",
        )
    ),
    "app://*",
    "file://*",
    "tauri://*",
    "vscode-webview://*",
    "vscode-file://*",
)

CONTENT_TYPES: dict[AudioFormat, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

MIME_SUFFIXES = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/flac": "flac",
    "audio/ogg": "ogg",
    "audio/opus": "opus",
    "audio/aac": "aac",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
}

EMOTION_NAMES = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)


def parse_cors_origins(value: str | None) -> tuple[str, ...]:
    """Return Ollama-style comma-separated origin patterns plus safe local defaults."""
    if value is None:
        return DEFAULT_CORS_ORIGINS
    unquoted = value.strip().strip("\"'")
    configured = tuple(pattern.strip() for pattern in unquoted.split(",") if pattern.strip())
    return configured + DEFAULT_CORS_ORIGINS


def origin_matches(origin: str, patterns: Sequence[str]) -> bool:
    return any(
        re.fullmatch(re.escape(pattern).replace(r"\*", ".*"), origin, flags=re.IGNORECASE) for pattern in patterns
    )


class OriginPatternCORSMiddleware(CORSMiddleware):
    """Starlette CORS middleware with Ollama-style wildcard origin matching."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        allow_origin_patterns: Sequence[str],
        allow_methods: Sequence[str] = ("GET",),
        allow_headers: Sequence[str] = (),
        allow_private_network: bool = False,
    ) -> None:
        self.allow_origin_patterns = tuple(allow_origin_patterns)
        super().__init__(
            app,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
            allow_private_network=allow_private_network,
        )

    def is_allowed_origin(self, origin: str) -> bool:
        return origin_matches(origin, self.allow_origin_patterns)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NamedAudioReference(StrictModel):
    id: str = Field(min_length=1)


class InlineAudioReference(StrictModel):
    audio: str = Field(min_length=1)
    format: ReferenceAudioFormat | None = None


AudioReference = Annotated[str, Field(min_length=1)] | NamedAudioReference | InlineAudioReference


class SameAsVoiceEmotion(StrictModel):
    mode: Literal["same_as_voice"]


class AudioEmotion(StrictModel):
    mode: Literal["audio"]
    reference: AudioReference
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class VectorEmotion(StrictModel):
    mode: Literal["vector"]
    values: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(min_length=8, max_length=8)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    randomize: bool = False


class TextEmotion(StrictModel):
    mode: Literal["text"]
    text: str = Field(min_length=1, max_length=4096)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


Emotion = Annotated[
    SameAsVoiceEmotion | AudioEmotion | VectorEmotion | TextEmotion,
    Field(discriminator="mode"),
]


class GenerationOptions(StrictModel):
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1, le=200)
    num_beams: int | None = Field(default=None, ge=1, le=20)
    repetition_penalty: float | None = Field(default=None, gt=0.0, le=100.0)


class ExtraOptions(StrictModel):
    language: Language = "auto"
    text_normalization: bool = True
    interval_silence_ms: int = Field(default=200, ge=0, le=5000)
    max_text_tokens_per_segment: int = Field(default=120, ge=1, le=1000)
    emotion: Emotion | None = None
    generation: GenerationOptions = Field(default_factory=GenerationOptions)


class SpeechRequest(StrictModel):
    model: str = Field(min_length=1)
    input: str = Field(min_length=1, max_length=4096)
    voice: AudioReference
    instructions: str | None = Field(default=None, min_length=1, max_length=4096)
    response_format: AudioFormat = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream_format: Literal["audio"] = "audio"
    extra: ExtraOptions = Field(default_factory=ExtraOptions)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    id: str
    version: ModelVersion
    model_dir: Path


@dataclass(frozen=True, slots=True)
class ServerConfig:
    models: dict[str, ModelConfig]
    voices: dict[str, Path]
    default_language: Language = "auto"
    device: str | None = None
    use_bf16: bool = False
    use_cuda_kernel: bool | None = None
    use_deepspeed: bool = False
    use_accel: bool = False
    use_torch_compile: bool = False
    use_qwen_emo: bool = False
    max_reference_audio_bytes: int = 20 * 1024 * 1024
    reference_cache_size: int = 32
    api_key: str | None = field(default=None, repr=False)
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("at least one model must be configured")
        if self.max_reference_audio_bytes <= 0:
            raise ValueError("max_reference_audio_bytes must be greater than zero")
        if self.reference_cache_size <= 0:
            raise ValueError("reference_cache_size must be greater than zero")


class Synthesizer(Protocol):
    def load(self) -> None: ...

    def synthesize(
        self,
        *,
        text: str,
        voice: Path,
        output_format: AudioFormat,
        speed: float,
        extra: ExtraOptions,
        emotion: Emotion | None,
        emotion_audio: Path | None,
    ) -> bytes: ...


class APIError(HTTPException):
    def __init__(
        self,
        status_code: int,
        detail: str,
        *,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_type = error_type
        self.param = param
        self.code = code


class ReferenceAudioStore:
    """Resolves named or inline audio to stable paths for IndexTTS cache reuse."""

    def __init__(self, config: ServerConfig) -> None:
        self._voices = config.voices
        self._max_bytes = config.max_reference_audio_bytes
        self._cache_size = config.reference_cache_size
        self._temp_dir = tempfile.TemporaryDirectory(prefix="indextts-references-")
        self._cache: OrderedDict[str, Path] = OrderedDict()
        self._lock = threading.Lock()

    def close(self) -> None:
        self._temp_dir.cleanup()

    def resolve(self, reference: AudioReference, *, param: str) -> Path:
        if isinstance(reference, NamedAudioReference):
            return self._resolve_name(reference.id, param=param)
        if isinstance(reference, InlineAudioReference):
            return self._resolve_inline(reference.audio, reference.format, param=param)
        if reference.startswith("data:audio/"):
            return self._resolve_inline(reference, None, param=param)
        return self._resolve_name(reference, param=param)

    def _resolve_name(self, name: str, *, param: str) -> Path:
        path = self._voices.get(name)
        if path is None:
            available = ", ".join(sorted(self._voices)) or "none"
            raise APIError(
                400,
                f"Unknown voice '{name}'. Available voices: {available}",
                param=param,
                code="voice_not_found",
            )
        return path

    def _resolve_inline(
        self,
        value: str,
        declared_format: ReferenceAudioFormat | None,
        *,
        param: str,
    ) -> Path:
        encoded = value
        audio_format = declared_format
        if value.startswith("data:"):
            metadata, separator, encoded = value.partition(",")
            if not separator or not metadata.endswith(";base64"):
                raise APIError(400, "voice audio must be a base64 Data URL", param=param)
            mime_type = metadata[5:].split(";", 1)[0].lower()
            audio_format = MIME_SUFFIXES.get(mime_type)
            if audio_format is None:
                raise APIError(400, f"Unsupported reference audio MIME type: {mime_type}", param=param)
        elif audio_format is None:
            raise APIError(
                400,
                "format is required when voice.audio contains raw base64",
                param=param,
            )

        compact = "".join(encoded.split())
        estimated_size = len(compact) * 3 // 4
        if estimated_size > self._max_bytes:
            raise APIError(
                413,
                f"Reference audio exceeds the {self._max_bytes}-byte limit",
                param=param,
                code="reference_audio_too_large",
            )
        try:
            data = base64.b64decode(compact, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise APIError(400, "Reference audio is not valid base64", param=param) from exc
        if not data:
            raise APIError(400, "Reference audio is empty", param=param)
        if len(data) > self._max_bytes:
            raise APIError(
                413,
                f"Reference audio exceeds the {self._max_bytes}-byte limit",
                param=param,
                code="reference_audio_too_large",
            )

        digest = hashlib.sha256(data).hexdigest()
        with self._lock:
            cached = self._cache.get(digest)
            if cached is not None:
                self._cache.move_to_end(digest)
                return cached
            path = Path(self._temp_dir.name) / f"{digest}.{audio_format}"
            path.write_bytes(data)
            try:
                _validate_audio_file(path, param=param)
            except Exception:
                path.unlink(missing_ok=True)
                raise
            self._cache[digest] = path
            while len(self._cache) > self._cache_size:
                _, evicted = self._cache.popitem(last=False)
                evicted.unlink(missing_ok=True)
            return path


class IndexTTSSynthesizer:
    """Owns one IndexTTS model and serializes access to its mutable caches."""

    def __init__(self, model_config: ModelConfig, server_config: ServerConfig) -> None:
        self._model_config = model_config
        self._server_config = server_config
        self._model = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if self._model_config.version == "2.5":
                from indextts.infer_v2_5 import IndexTTS2

                precision = {"use_bf16": self._server_config.use_bf16}
            else:
                from indextts.infer_v2 import IndexTTS2

                precision = {"use_fp16": self._server_config.use_bf16}
            model_dir = self._model_config.model_dir
            self._model = IndexTTS2(
                cfg_path=str(model_dir / "config.yaml"),
                model_dir=str(model_dir),
                device=self._server_config.device,
                use_cuda_kernel=self._server_config.use_cuda_kernel,
                use_deepspeed=self._server_config.use_deepspeed,
                use_accel=self._server_config.use_accel,
                use_torch_compile=self._server_config.use_torch_compile,
                use_qwen_emo=self._server_config.use_qwen_emo,
                **precision,
            )

    def synthesize(
        self,
        *,
        text: str,
        voice: Path,
        output_format: AudioFormat,
        speed: float,
        extra: ExtraOptions,
        emotion: Emotion | None,
        emotion_audio: Path | None,
    ) -> bytes:
        self.load()
        assert self._model is not None
        with self._inference_lock, tempfile.TemporaryDirectory(prefix="indextts-speech-") as temp_dir:
            wav_path = Path(temp_dir) / "speech.wav"
            infer_kwargs = self._infer_kwargs(
                text=text,
                voice=voice,
                wav_path=wav_path,
                extra=extra,
                emotion=emotion,
                emotion_audio=emotion_audio,
                speed=speed,
            )
            result = self._model.infer(**infer_kwargs)
            if result is None or not wav_path.is_file():
                raise RuntimeError("IndexTTS did not produce an audio file")
            postprocess_speed = speed if self._model_config.version == "2" else 1.0
            return encode_audio(wav_path, output_format, speed=postprocess_speed)

    def _infer_kwargs(
        self,
        *,
        text: str,
        voice: Path,
        wav_path: Path,
        extra: ExtraOptions,
        emotion: Emotion | None,
        emotion_audio: Path | None,
        speed: float,
    ) -> dict:
        kwargs = {
            "spk_audio_prompt": str(voice),
            "text": text,
            "output_path": str(wav_path),
            "interval_silence": extra.interval_silence_ms,
            "max_text_tokens_per_segment": extra.max_text_tokens_per_segment,
            "verbose": False,
            **extra.generation.model_dump(exclude_none=True),
        }
        if isinstance(emotion, AudioEmotion):
            kwargs.update(emo_audio_prompt=str(emotion_audio), emo_alpha=emotion.weight)
        elif isinstance(emotion, VectorEmotion):
            kwargs.update(
                emo_vector=emotion.values,
                emo_alpha=emotion.weight,
                use_random=emotion.randomize,
            )
        elif isinstance(emotion, TextEmotion):
            kwargs.update(use_emo_text=True, emo_text=emotion.text, emo_alpha=emotion.weight)

        if self._model_config.version == "2.5":
            kwargs.update(
                lang=extra.language,
                duration_factor=1.0 / speed,
                text_normalization=extra.text_normalization,
            )
        return kwargs


def resolve_model_dir(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if (candidate / "config.yaml").is_file():
        return candidate
    checkpoints = candidate / "checkpoints"
    if (checkpoints / "config.yaml").is_file():
        return checkpoints
    raise ValueError(f"model directory must contain config.yaml directly or under checkpoints/: {candidate}")


def detect_language(text: str) -> Language:
    if any("\u3040" <= char <= "\u30ff" for char in text):
        return "ja"
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "zh"
    if any("\uac00" <= char <= "\ud7af" for char in text):
        return "ko"
    if any("\u0400" <= char <= "\u04ff" for char in text):
        return "ru"
    if any("\u0600" <= char <= "\u06ff" for char in text):
        return "ar"
    return "en"


def _validate_audio_file(path: Path, *, param: str) -> None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to validate inline reference audio")
    process = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        check=False,
    )
    if process.returncode != 0 or not process.stdout.strip():
        raise APIError(400, "Reference audio could not be decoded", param=param)


def _atempo_filter(speed: float) -> str:
    factors = []
    remaining = speed
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    factors.append(remaining)
    return ",".join(f"atempo={factor:g}" for factor in factors)


def encode_audio(wav_path: Path, output_format: AudioFormat, *, speed: float = 1.0) -> bytes:
    if output_format == "wav" and speed == 1.0:
        return wav_path.read_bytes()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for audio conversion")
    output_args: dict[AudioFormat, list[str]] = {
        "mp3": ["-f", "mp3"],
        "opus": ["-c:a", "libopus", "-f", "ogg"],
        "aac": ["-c:a", "aac", "-f", "adts"],
        "flac": ["-f", "flac"],
        "pcm": ["-c:a", "pcm_s16le", "-f", "s16le"],
        "wav": ["-f", "wav"],
    }
    filter_args = ["-filter:a", _atempo_filter(speed)] if speed != 1.0 else []
    with tempfile.TemporaryDirectory(prefix="indextts-encode-") as temp_dir:
        output_path = Path(temp_dir) / f"speech.{output_format}"
        process = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                str(wav_path),
                *filter_args,
                *output_args[output_format],
                "-y",
                str(output_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.returncode != 0:
            detail = process.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to encode {output_format}: {detail}")
        return output_path.read_bytes()


def _error(
    message: str,
    *,
    error_type: str,
    param: str | None = None,
    code: str | None = None,
) -> dict[str, dict[str, str | None]]:
    return {"error": {"message": message, "type": error_type, "param": param, "code": code}}


def create_app(
    config: ServerConfig,
    synthesizers: Mapping[str, Synthesizer] | None = None,
) -> FastAPI:
    engines = (
        dict(synthesizers)
        if synthesizers is not None
        else {model_id: IndexTTSSynthesizer(model_config, config) for model_id, model_config in config.models.items()}
    )
    references = ReferenceAudioStore(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        for engine in engines.values():
            await run_in_threadpool(engine.load)
        try:
            yield
        finally:
            references.close()

    app = FastAPI(title="IndexTTS OpenAI-compatible server", lifespan=lifespan)
    app.add_middleware(
        OriginPatternCORSMiddleware,
        allow_origin_patterns=config.cors_origins,
        allow_methods=("GET", "POST", "OPTIONS"),
        allow_headers=("*",),
        allow_private_network=True,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        location = first.get("loc", ())
        param = ".".join(str(part) for part in location[1:]) or None
        return JSONResponse(
            status_code=422,
            content=_error(first["msg"], error_type="invalid_request_error", param=param),
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
        error_type = getattr(
            exc,
            "error_type",
            "server_error" if exc.status_code >= 500 else "invalid_request_error",
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error(
                str(exc.detail),
                error_type=error_type,
                param=getattr(exc, "param", None),
                code=getattr(exc, "code", None),
            ),
            headers=exc.headers,
        )

    def authorize(authorization: str | None) -> None:
        if config.api_key is None:
            return
        expected = f"Bearer {config.api_key}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise APIError(
                401,
                "Incorrect API key provided",
                code="invalid_api_key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "models": sorted(config.models),
            "voices": sorted(config.voices),
        }

    @app.get("/v1/models")
    async def list_models(authorization: Annotated[str | None, Header()] = None) -> dict[str, object]:
        authorize(authorization)
        return {
            "object": "list",
            "data": [
                {
                    "id": model.id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                    "capabilities": {
                        "emotion_audio": True,
                        "emotion_vector": True,
                        "emotion_text": config.use_qwen_emo,
                        "language": model.version == "2.5",
                        "emotion_vector_order": list(EMOTION_NAMES),
                    },
                }
                for model in config.models.values()
            ],
        }

    @app.post("/v1/audio/speech", response_class=Response)
    async def create_speech(
        speech: SpeechRequest,
        authorization: Annotated[str | None, Header()] = None,
    ) -> Response:
        authorize(authorization)
        model_config = config.models.get(speech.model)
        engine = engines.get(speech.model)
        if model_config is None or engine is None:
            raise APIError(
                404,
                f"The model '{speech.model}' does not exist or is not loaded",
                param="model",
                code="model_not_found",
            )

        emotion = speech.extra.emotion
        if speech.instructions is not None:
            if emotion is not None:
                raise APIError(
                    400,
                    "instructions and extra.emotion cannot be used together",
                    param="instructions",
                )
            emotion = TextEmotion(mode="text", text=speech.instructions)
        if isinstance(emotion, TextEmotion) and not config.use_qwen_emo:
            raise APIError(
                400,
                "Text emotion requires the server to start with --qwen-emo",
                param="extra.emotion",
                code="unsupported_parameter",
            )
        if model_config.version == "2" and speech.extra.language != "auto":
            raise APIError(
                400,
                "extra.language is only supported by index-tts-2.5",
                param="extra.language",
                code="unsupported_parameter",
            )
        if model_config.version == "2" and not speech.extra.text_normalization:
            raise APIError(
                400,
                "extra.text_normalization is only supported by index-tts-2.5",
                param="extra.text_normalization",
                code="unsupported_parameter",
            )

        if speech.extra.language == "auto":
            configured = config.default_language
            speech.extra.language = detect_language(speech.input) if configured == "auto" else configured
        voice = await run_in_threadpool(references.resolve, speech.voice, param="voice")
        emotion_audio = None
        if isinstance(emotion, AudioEmotion):
            emotion_audio = await run_in_threadpool(
                references.resolve,
                emotion.reference,
                param="extra.emotion.reference",
            )
        try:
            audio = await run_in_threadpool(
                engine.synthesize,
                text=speech.input,
                voice=voice,
                output_format=speech.response_format,
                speed=speech.speed,
                extra=speech.extra,
                emotion=emotion,
                emotion_audio=emotion_audio,
            )
        except APIError:
            raise
        except Exception as exc:
            raise APIError(500, str(exc), error_type="server_error") from exc
        return Response(
            content=audio,
            media_type=CONTENT_TYPES[speech.response_format],
            headers={"Content-Disposition": f'inline; filename="speech.{speech.response_format}"'},
        )

    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name
    return app
