"""OpenAI-compatible HTTP server for IndexTTS."""

from inference_index_tts.server import ModelConfig, ServerConfig, create_app

__all__ = ["ModelConfig", "ServerConfig", "create_app"]
