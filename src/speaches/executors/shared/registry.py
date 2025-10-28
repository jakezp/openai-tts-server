from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from speaches.config import Config

from speaches.executors.kokoro import KokoroModelManager, kokoro_model_registry
from speaches.executors.parakeet import ParakeetModelManager, parakeet_model_registry
from speaches.executors.piper import PiperModelManager, piper_model_registry
from speaches.executors.pyannote import PyannoteModelManager, pyannote_model_registry
from speaches.executors.shared.executor import Executor
from speaches.executors.whisper import WhisperModelManager, whisper_model_registry
from speaches.executors.styletts2.registry import discover_preset_voices, PRESET_VOICES


class ExecutorRegistry:
    def __init__(self, config: Config) -> None:
        self._whisper_executor = Executor(
            name="whisper",
            model_manager=WhisperModelManager(config.stt_model_ttl, config.whisper),
            model_registry=whisper_model_registry,
            task="automatic-speech-recognition",
        )
        self._parakeet_executor = Executor(
            name="parakeet",
            model_manager=ParakeetModelManager(config.stt_model_ttl, config.unstable_ort_opts),
            model_registry=parakeet_model_registry,
            task="automatic-speech-recognition",
        )
        self._piper_executor = Executor(
            name="piper",
            model_manager=PiperModelManager(config.tts_model_ttl, config.unstable_ort_opts),
            model_registry=piper_model_registry,
            task="text-to-speech",
        )
        self._kokoro_executor = Executor(
            name="kokoro",
            model_manager=KokoroModelManager(config.tts_model_ttl, config.unstable_ort_opts),
            model_registry=kokoro_model_registry,
            task="text-to-speech",
        )
        
        # Initialize StyleTTS2 executor
        from speaches.executors.styletts2.model_manager import StyleTTS2ModelManager
        from speaches.executors.styletts2.registry import styletts2_model_registry
        # Ensure PRESET_VOICES is populated by scanning the voices directory
        voices_dir = Path(__file__).parent.parent.parent.parent.parent / "voices"
        try:
            discovered = discover_preset_voices(voices_dir)
            PRESET_VOICES.clear()
            PRESET_VOICES.extend(discovered)
        except Exception:
            import logging
            logger = logging.getLogger(__name__)
            logger.exception("Failed to discover preset voices")
        self._styletts2_executor = Executor(
            name="styletts2",
            model_manager=StyleTTS2ModelManager(config=config, voices_dir=voices_dir),
            model_registry=styletts2_model_registry,
            task="text-to-speech",
        )
        
        self._pyannote_executor = Executor(
            name="pyannote",
            model_manager=PyannoteModelManager(config.stt_model_ttl, config.unstable_ort_opts),
            model_registry=pyannote_model_registry,
            task="speaker-embedding",
        )

    @property
    def transcription(self):  # noqa: ANN201
        return (self._whisper_executor, self._parakeet_executor)

    @property
    def text_to_speech(self):  # noqa: ANN201
        return (self._piper_executor, self._kokoro_executor, self._styletts2_executor)

    @property
    def speaker_embedding(self):  # noqa: ANN201
        return (self._pyannote_executor,)

    def all_executors(self):  # noqa: ANN201
        return (
            self._whisper_executor,
            self._parakeet_executor,
            self._piper_executor,
            self._kokoro_executor,
            self._styletts2_executor,
            self._pyannote_executor,
        )
