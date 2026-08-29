from __future__ import annotations

import asyncio
import base64
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Any, cast

from inference_index_tts.server import (
    AudioEmotion,
    AudioFormat,
    IndexTTSSynthesizer,
    ModelConfig,
    ServerConfig,
    TextEmotion,
    VectorEmotion,
    create_app,
    detect_language,
    encode_audio,
    origin_matches,
    parse_cors_origins,
    resolve_model_dir,
)


class FakeSynthesizer:
    def __init__(self, result: bytes = b"encoded audio") -> None:
        self.calls = []
        self.result = result

    def load(self) -> None:
        pass

    def synthesize(self, **kwargs) -> bytes:
        self.calls.append(kwargs)
        return self.result


def wav_bytes(frame_count: int = 2205) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(22050)
        output.writeframes(b"\x00\x00" * frame_count)
    return buffer.getvalue()


async def request(
    app,
    body: dict | None = None,
    *,
    path: str = "/v1/audio/speech",
    method: str = "POST",
    authorization: str | None = None,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
):
    payload = json.dumps(body).encode() if body is not None else b""
    sent = []
    headers = [(b"content-type", b"application/json")]
    if authorization is not None:
        headers.append((b"authorization", authorization.encode()))
    if extra_headers is not None:
        headers.extend(extra_headers)
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": payload, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": headers,
            "client": ("test", 1),
            "server": ("test", 80),
        },
        receive,
        send,
    )
    start = next(item for item in sent if item["type"] == "http.response.start")
    response_body = b"".join(item.get("body", b"") for item in sent if item["type"] == "http.response.body")
    return start, response_body


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.voice = self.root / "voice.wav"
        self.voice.write_bytes(wav_bytes())
        self.models = {
            "index-tts-2": ModelConfig("index-tts-2", "2", self.root),
            "index-tts-2.5": ModelConfig("index-tts-2.5", "2.5", self.root),
        }
        self.engines = {name: FakeSynthesizer() for name in self.models}
        self.config = ServerConfig(models=self.models, voices={"alice": self.voice})
        self.app = create_app(self.config, synthesizers=self.engines)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def speech_body(self, **updates):
        body = {"model": "index-tts-2.5", "input": "你好，世界", "voice": "alice"}
        body.update(updates)
        return body

    def test_localhost_cors_preflight_is_allowed_on_any_port(self):
        start, _ = asyncio.run(
            request(
                self.app,
                path="/v1/audio/speech",
                method="OPTIONS",
                extra_headers=[
                    (b"origin", b"http://localhost:5173"),
                    (b"access-control-request-method", b"POST"),
                    (b"access-control-request-headers", b"authorization,content-type"),
                    (b"access-control-request-private-network", b"true"),
                ],
            )
        )
        headers = dict(start["headers"])
        self.assertEqual(start["status"], 200)
        self.assertEqual(headers[b"access-control-allow-origin"], b"http://localhost:5173")
        self.assertIn(b"authorization", headers[b"access-control-allow-headers"].lower())
        self.assertEqual(headers[b"access-control-allow-private-network"], b"true")

    def test_cors_origin_patterns_include_configured_and_loopback_origins(self):
        patterns = parse_cors_origins(" https://*.example.com,chrome-extension://abc123 ")
        self.assertTrue(origin_matches("https://studio.example.com", patterns))
        self.assertTrue(origin_matches("chrome-extension://abc123", patterns))
        self.assertTrue(origin_matches("https://127.0.0.1:8443", patterns))
        self.assertTrue(origin_matches("http://[::1]:3000", patterns))
        self.assertFalse(origin_matches("https://example.net", patterns))

    def test_unconfigured_remote_cors_origin_is_rejected(self):
        start, _ = asyncio.run(
            request(
                self.app,
                path="/v1/audio/speech",
                method="OPTIONS",
                extra_headers=[
                    (b"origin", b"https://example.net"),
                    (b"access-control-request-method", b"POST"),
                ],
            )
        )
        self.assertEqual(start["status"], 400)
        self.assertNotIn(b"access-control-allow-origin", dict(start["headers"]))

    def test_request_model_routes_to_registered_engine(self):
        start, body = asyncio.run(request(self.app, self.speech_body(response_format="wav")))
        self.assertEqual(start["status"], 200)
        self.assertEqual(body, b"encoded audio")
        self.assertEqual(len(self.engines["index-tts-2.5"].calls), 1)
        self.assertEqual(len(self.engines["index-tts-2"].calls), 0)
        call = self.engines["index-tts-2.5"].calls[0]
        self.assertEqual(call["extra"].language, "zh")
        self.assertEqual(call["voice"], self.voice)

    def test_unregistered_model_returns_model_not_found(self):
        start, body = asyncio.run(request(self.app, self.speech_body(model="tts-1")))
        error = json.loads(body)["error"]
        self.assertEqual(start["status"], 404)
        self.assertEqual(error["code"], "model_not_found")
        self.assertEqual(error["param"], "model")

    def test_models_endpoint_lists_only_registered_models(self):
        start, body = asyncio.run(request(self.app, path="/v1/models", method="GET"))
        self.assertEqual(start["status"], 200)
        self.assertEqual(
            {item["id"] for item in json.loads(body)["data"]},
            {"index-tts-2", "index-tts-2.5"},
        )

    def test_inline_data_url_voice_is_decoded_and_cached(self):
        data_url = "data:audio/wav;base64," + base64.b64encode(wav_bytes()).decode()
        body = self.speech_body(voice={"audio": data_url})
        first, _ = asyncio.run(request(self.app, body))
        second, _ = asyncio.run(request(self.app, body))
        self.assertEqual(first["status"], 200)
        self.assertEqual(second["status"], 200)
        calls = self.engines["index-tts-2.5"].calls
        self.assertEqual(calls[-2]["voice"], calls[-1]["voice"])
        self.assertTrue(calls[-1]["voice"].is_file())

    def test_raw_base64_voice_with_format_is_accepted(self):
        encoded = base64.b64encode(wav_bytes()).decode()
        start, _ = asyncio.run(request(self.app, self.speech_body(voice={"audio": encoded, "format": "wav"})))
        self.assertEqual(start["status"], 200)

    def test_same_as_voice_emotion_is_explicitly_supported(self):
        start, _ = asyncio.run(
            request(
                self.app,
                self.speech_body(extra={"emotion": {"mode": "same_as_voice"}}),
            )
        )
        self.assertEqual(start["status"], 200)
        emotion = self.engines["index-tts-2.5"].calls[-1]["emotion"]
        self.assertEqual(emotion.mode, "same_as_voice")

    def test_raw_base64_voice_requires_format(self):
        encoded = base64.b64encode(wav_bytes()).decode()
        start, body = asyncio.run(request(self.app, self.speech_body(voice={"audio": encoded})))
        self.assertEqual(start["status"], 400)
        self.assertEqual(json.loads(body)["error"]["param"], "voice")

    def test_audio_emotion_resolves_registered_reference(self):
        start, _ = asyncio.run(
            request(
                self.app,
                self.speech_body(
                    extra={
                        "emotion": {
                            "mode": "audio",
                            "reference": {"id": "alice"},
                            "weight": 0.7,
                        }
                    }
                ),
            )
        )
        self.assertEqual(start["status"], 200)
        call = self.engines["index-tts-2.5"].calls[-1]
        self.assertIsInstance(call["emotion"], AudioEmotion)
        self.assertEqual(call["emotion_audio"], self.voice)
        self.assertEqual(call["emotion"].weight, 0.7)

    def test_vector_emotion(self):
        values = [0.8, 0, 0, 0, 0, 0, 0, 0]
        start, _ = asyncio.run(
            request(
                self.app,
                self.speech_body(
                    extra={
                        "emotion": {
                            "mode": "vector",
                            "values": values,
                            "weight": 0.5,
                            "randomize": True,
                        }
                    }
                ),
            )
        )
        self.assertEqual(start["status"], 200)
        emotion = self.engines["index-tts-2.5"].calls[-1]["emotion"]
        self.assertIsInstance(emotion, VectorEmotion)
        self.assertEqual(emotion.values, values)

    def test_text_emotion_requires_qwen(self):
        start, body = asyncio.run(
            request(
                self.app,
                self.speech_body(extra={"emotion": {"mode": "text", "text": "平静", "weight": 0.8}}),
            )
        )
        self.assertEqual(start["status"], 400)
        self.assertEqual(json.loads(body)["error"]["code"], "unsupported_parameter")

    def test_instructions_map_to_text_emotion_when_qwen_is_enabled(self):
        config = ServerConfig(models=self.models, voices={"alice": self.voice}, use_qwen_emo=True)
        app = create_app(config, synthesizers=self.engines)
        start, _ = asyncio.run(request(app, self.speech_body(instructions="温柔而平静")))
        self.assertEqual(start["status"], 200)
        emotion = self.engines["index-tts-2.5"].calls[-1]["emotion"]
        self.assertIsInstance(emotion, TextEmotion)
        self.assertEqual(emotion.text, "温柔而平静")

    def test_instructions_conflict_with_explicit_emotion(self):
        config = ServerConfig(models=self.models, voices={"alice": self.voice}, use_qwen_emo=True)
        app = create_app(config, synthesizers=self.engines)
        start, body = asyncio.run(
            request(
                app,
                self.speech_body(
                    instructions="温柔",
                    extra={"emotion": {"mode": "same_as_voice"}},
                ),
            )
        )
        self.assertEqual(start["status"], 400)
        self.assertEqual(json.loads(body)["error"]["param"], "instructions")

    def test_v2_rejects_explicit_language(self):
        body = self.speech_body(model="index-tts-2", extra={"language": "zh"})
        start, response_body = asyncio.run(request(self.app, body))
        self.assertEqual(start["status"], 400)
        self.assertEqual(json.loads(response_body)["error"]["param"], "extra.language")

    def test_validation_uses_openai_error_shape(self):
        body = self.speech_body(extra={"emotion": {"mode": "vector", "values": [0.5, 0.5]}})
        start, response_body = asyncio.run(request(self.app, body))
        self.assertEqual(start["status"], 422)
        self.assertTrue(json.loads(response_body)["error"]["param"].startswith("extra.emotion"))

    def test_optional_api_key(self):
        config = ServerConfig(models=self.models, voices={"alice": self.voice}, api_key="secret")
        app = create_app(config, synthesizers=self.engines)
        denied, _ = asyncio.run(request(app, self.speech_body()))
        allowed, _ = asyncio.run(request(app, self.speech_body(), authorization="Bearer secret"))
        self.assertEqual(denied["status"], 401)
        self.assertEqual(allowed["status"], 200)

    def test_language_detection(self):
        self.assertEqual(detect_language("English text"), "en")
        self.assertEqual(detect_language("中文"), "zh")
        self.assertEqual(detect_language("日本語です"), "ja")
        self.assertEqual(detect_language("한국어"), "ko")

    def test_model_dir_accepts_parent_or_checkpoints(self):
        checkpoints = self.root / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "config.yaml").touch()
        self.assertEqual(resolve_model_dir(self.root), checkpoints)
        self.assertEqual(resolve_model_dir(checkpoints), checkpoints)

    def test_ffmpeg_encodes_supported_formats_and_v2_speed(self):
        wav_path = self.root / "silence.wav"
        wav_path.write_bytes(wav_bytes())
        signatures: dict[AudioFormat, tuple[bytes, ...]] = {
            "mp3": (b"ID3", b"\xff"),
            "opus": (b"OggS",),
            "aac": (b"\xff",),
            "flac": (b"fLaC",),
        }
        for output_format, expected in signatures.items():
            with self.subTest(output_format=output_format):
                encoded = encode_audio(wav_path, output_format)
                self.assertGreater(len(encoded), 0)
                self.assertTrue(encoded.startswith(expected))
        slowed_wav = encode_audio(wav_path, "wav", speed=0.5)
        with wave.open(io.BytesIO(slowed_wav)) as audio:
            self.assertGreater(audio.getnframes(), 2500)

    def test_v25_adapter_passes_language_speed_emotion_and_generation(self):
        class FakeModel:
            def __init__(self):
                self.kwargs: dict[str, Any] | None = None

            def infer(self, **kwargs):
                self.kwargs = kwargs
                Path(kwargs["output_path"]).write_bytes(wav_bytes())
                return kwargs["output_path"]

        model = FakeModel()
        adapter = IndexTTSSynthesizer(self.models["index-tts-2.5"], self.config)
        adapter._model = cast(Any, model)
        extra = self._extra(
            language="zh",
            generation={"temperature": 0.6, "top_p": 0.7},
        )
        emotion = VectorEmotion(mode="vector", values=[0.8, 0, 0, 0, 0, 0, 0, 0])
        adapter.synthesize(
            text="hello",
            voice=self.voice,
            output_format="wav",
            speed=2,
            extra=extra,
            emotion=emotion,
            emotion_audio=None,
        )
        assert model.kwargs is not None
        self.assertEqual(model.kwargs["lang"], "zh")
        self.assertEqual(model.kwargs["duration_factor"], 0.5)
        self.assertEqual(model.kwargs["emo_vector"], emotion.values)
        self.assertEqual(model.kwargs["temperature"], 0.6)

    def test_v2_adapter_omits_v25_arguments_and_postprocesses_speed(self):
        class FakeModel:
            def __init__(self):
                self.kwargs: dict[str, Any] | None = None

            def infer(self, **kwargs):
                self.kwargs = kwargs
                Path(kwargs["output_path"]).write_bytes(wav_bytes())
                return kwargs["output_path"]

        model = FakeModel()
        adapter = IndexTTSSynthesizer(self.models["index-tts-2"], self.config)
        adapter._model = cast(Any, model)
        audio = adapter.synthesize(
            text="hello",
            voice=self.voice,
            output_format="wav",
            speed=2,
            extra=self._extra(),
            emotion=AudioEmotion(mode="audio", reference="alice", weight=0.6),
            emotion_audio=self.voice,
        )
        assert model.kwargs is not None
        self.assertNotIn("lang", model.kwargs)
        self.assertNotIn("duration_factor", model.kwargs)
        self.assertNotIn("text_normalization", model.kwargs)
        self.assertEqual(model.kwargs["emo_audio_prompt"], str(self.voice))
        self.assertEqual(model.kwargs["emo_alpha"], 0.6)
        with wave.open(io.BytesIO(audio)) as output:
            self.assertLess(output.getnframes(), 1500)

    @staticmethod
    def _extra(**values):
        from inference_index_tts.server import ExtraOptions

        return ExtraOptions.model_validate(values)


if __name__ == "__main__":
    unittest.main()
