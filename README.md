# inference-index-tts

Expose local IndexTTS 2 and 2.5 models through an OpenAI-compatible TTS endpoint powered by FastAPI.

## Start the server

Use the repeatable `--model 'NAME@file://PATH'` option to register the models available to the server. A model URI may point to either a `checkpoints` directory or its parent directory.

```bash
pixi install

pixi run serve \
  --model 'index-tts-2@file:///models/index-tts-2/checkpoints' \
  --model 'index-tts-2.5@file:///models/index-tts-2.5/checkpoints' \
  --voice 'alice@file:///voices/alice.wav' \
  --voice 'bob@file:///voices/bob.wav' \
  --device cuda:0 \
  --half \
  --host 0.0.0.0 \
  --port 8000
```

Registering only one model is also supported. The request's `model` must be an `index-tts-2` or `index-tts-2.5` model registered at startup. Use `GET /v1/models` to list available models and `GET /health` to check the server status.

`file:///absolute/path` represents an absolute path. `file://relative/path` is resolved relative to the server's working directory. Named voices are optional because requests can provide reference audio directly.

Common server options:

- `--qwen-emo`: load QwenEmotion and enable `instructions` and text-based emotion control.
- `--api-key SECRET`: require clients to provide a bearer token.
- `--max-reference-audio-bytes`: set the maximum decoded size of inline reference audio. The default is 20 MiB.
- `--reference-cache-size`: set the number of inline reference audio files cached by content. The default is 32.
- `--default-language`: set the language used when a request specifies `extra.language=auto`.

### Browser origins (CORS)

Following Ollama's origin configuration model, HTTP and HTTPS origins on `localhost`, `127.0.0.1`, `0.0.0.0`, and `[::1]` are allowed by default on any port. Additional origins can be supplied as a comma-separated list in `INFERENCE_INDEX_TTS_ORIGINS`; `*` wildcards are supported:

```bash
INFERENCE_INDEX_TTS_ORIGINS='https://studio.example.com,chrome-extension://*' \
  pixi run serve --model 'index-tts-2.5@file:///models/index-tts-2.5/checkpoints'
```

Configured origins are added to the local defaults. CORS preflight requests accept the server's `GET` and `POST` methods and requested headers, including `Authorization`.

## Basic request

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "index-tts-2.5",
    "input": "Hello, this is IndexTTS.",
    "voice": "alice",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output speech.mp3
```

`voice` accepts any of the following forms:

```json
"alice"
```

```json
{"id": "alice"}
```

```json
"data:audio/wav;base64,UklGRi..."
```

```json
{"audio": "UklGRi...", "format": "wav"}
```

Reference audio may use WAV, MP3, FLAC, OGG, Opus, AAC, or M4A. Inline audio is subject to a decoded-size limit, validated with FFprobe, and cached by SHA-256 digest. Base64 content is never written to logs.

## Extra options

Place IndexTTS-specific options under `extra`:

```json
{
  "model": "index-tts-2.5",
  "input": "I am in a great mood today.",
  "voice": "alice",
  "response_format": "mp3",
  "speed": 1.1,
  "extra": {
    "language": "en",
    "text_normalization": true,
    "interval_silence_ms": 200,
    "max_text_tokens_per_segment": 120,
    "generation": {
      "temperature": 0.8,
      "top_p": 0.8,
      "top_k": 30,
      "num_beams": 3,
      "repetition_penalty": 10.0
    }
  }
}
```

`language` supports `auto`, `zh`, `en`, `ja`, `es`, `ar`, `ko`, and `ru`. Explicit language selection and `text_normalization=false` are supported only by `index-tts-2.5`.

For IndexTTS 2.5, `speed` maps to the native `duration_factor=1/speed` option. For IndexTTS 2, speed adjustment is applied afterward with FFmpeg's `atempo` filter.

## Emotion modes

When `extra.emotion` is omitted, the model uses the emotion from the voice reference audio. This is equivalent to:

```json
{"mode": "same_as_voice"}
```

Use separate reference audio for emotion:

```json
{
  "mode": "audio",
  "reference": {
    "audio": "data:audio/wav;base64,UklGRi..."
  },
  "weight": 0.8
}
```

`reference` may also be the name of a registered voice or an object such as `{"id":"alice"}`.

Use an eight-dimensional emotion vector:

```json
{
  "mode": "vector",
  "values": [0.8, 0, 0, 0, 0, 0, 0, 0],
  "weight": 1.0,
  "randomize": false
}
```

The vector order is fixed as `happy, angry, sad, afraid, disgusted, melancholic, surprised, calm`. Each value must be between 0 and 1.

Use a natural-language emotion description:

```json
{
  "mode": "text",
  "text": "Gentle and calm, with a hint of sadness",
  "weight": 0.8
}
```

Text mode requires the server to start with `--qwen-emo`. The top-level `instructions` field is a shortcut for text mode. A request cannot provide both `instructions` and an explicit `extra.emotion` value.

## Output formats

`response_format` supports `mp3`, `opus`, `aac`, `flac`, `wav`, and raw `pcm_s16le`. The endpoint currently returns regular audio responses and does not support `stream_format=sse`.

## Tests

```bash
pixi run test
```
