from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from inference_index_tts.cli import (
    REQUIRED_MODEL_FILES,
    parse_file_resource,
    parse_models,
    parse_voices,
)


class CliTests(unittest.TestCase):
    def test_parse_registered_voice_file_uri(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            voice = Path(temp_dir) / "my voice.wav"
            voice.touch()
            self.assertEqual(
                parse_voices([f"alice@{voice.as_uri()}"]),
                {"alice": voice.resolve()},
            )

    def test_parse_relative_file_uri(self):
        name, path = parse_file_resource(
            "alice@file://voices/reference.wav",
            resource="voice",
        )
        self.assertEqual(name, "alice")
        self.assertEqual(path, (Path.cwd() / "voices/reference.wav").resolve())

    def test_parse_models_routes_supported_versions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            v2 = root / "v2"
            v25 = root / "v25"
            v2.mkdir()
            v25.mkdir()
            for filename in REQUIRED_MODEL_FILES["2"]:
                (v2 / filename).touch()
            for filename in REQUIRED_MODEL_FILES["2.5"]:
                (v25 / filename).touch()
            models = parse_models(
                [
                    f"index-tts-2@{v2.as_uri()}",
                    f"index-tts-2.5@{v25.as_uri()}",
                ]
            )
            self.assertEqual(models["index-tts-2"].version, "2")
            self.assertEqual(models["index-tts-2.5"].version, "2.5")

    def test_rejects_unsupported_model(self):
        with self.assertRaisesRegex(ValueError, "unsupported model"):
            parse_models(["tts-1@file:///models/tts-1"])

    def test_rejects_old_mapping_syntax(self):
        with self.assertRaisesRegex(ValueError, "NAME@file://PATH"):
            parse_voices(["alloy=/voice.wav"])


if __name__ == "__main__":
    unittest.main()
