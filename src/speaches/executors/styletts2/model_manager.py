"""StyleTTS2 model manager with TTL-based lifecycle management."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from speaches.config import Config

from speaches.executors.shared.base_model_manager import BaseModelManager, SelfDisposingModel
from speaches.executors.styletts2.inference import StyleTTS2

logger = logging.getLogger(__name__)


class StyleTTS2ModelManager(BaseModelManager[StyleTTS2]):
    """Manager for StyleTTS2 models with automatic loading/unloading."""

    def __init__(self, config: Config, voices_dir: Path) -> None:
        """
        Initialize StyleTTS2 model manager.

        Args:
            config: Application configuration
            voices_dir: Directory containing voice reference files
        """
        super().__init__(ttl=config.tts_model_ttl)
        self.config = config
        self.voices_dir = voices_dir
        self.device = self._determine_device()
        self.voice_styles: dict[str, torch.Tensor] = {}  # Cache for voice embeddings

    def _determine_device(self) -> str:
        """Determine which device to use for inference."""
        if torch.cuda.is_available():
            logger.info("CUDA available, using GPU")
            return "cuda"
        # MPS support disabled for now due to compatibility
        # elif torch.backends.mps.is_available():
        #     logger.info("MPS available, using Apple Silicon GPU")
        #     return "mps"
        else:
            logger.info("No GPU available, using CPU")
            return "cpu"

    def _load_fn(self, model_id: str) -> StyleTTS2:
        """
        Create and initialize a StyleTTS2 model.

        Args:
            model_id: HuggingFace model repository ID

        Returns:
            Initialized StyleTTS2 model
        """
        logger.info(f"Loading StyleTTS2 model: {model_id}")
        model = StyleTTS2(model_id=model_id, device=self.device)  # type: ignore[arg-type]
        return model

    def get_voice_style(self, model: StyleTTS2, voice_name: str) -> torch.Tensor:
        """
        Get or compute voice style embedding.

        Args:
            model: StyleTTS2 model instance
            voice_name: Name of the voice (e.g., 'f-us-1', 'm-us-2')

        Returns:
            Voice style embedding tensor
        """
        cache_key = f"{model.model_id}:{voice_name}"

        if cache_key not in self.voice_styles:
            voice_path = self.voices_dir / f"{voice_name}.wav"
            if not voice_path.exists():
                raise FileNotFoundError(f"Voice file not found: {voice_path}")

            logger.info(f"Computing style for voice: {voice_name}")
            style = model.compute_style(voice_path)
            self.voice_styles[cache_key] = style

        return self.voice_styles[cache_key]

    def compute_custom_voice_style(self, model: StyleTTS2, audio_path: str | Path) -> torch.Tensor:
        """
        Compute style embedding from custom reference audio.

        Args:
            model: StyleTTS2 model instance
            audio_path: Path to custom reference audio file

        Returns:
            Voice style embedding tensor
        """
        logger.info(f"Computing custom voice style from: {audio_path}")
        return model.compute_style(audio_path)

    def on_model_unloaded(self, model_id: str) -> None:
        """
        Callback when a model is unloaded.

        Args:
            model_id: ID of the unloaded model
        """
        # Clear voice styles associated with this model
        keys_to_remove = [key for key in self.voice_styles if key.startswith(f"{model_id}:")]
        for key in keys_to_remove:
            del self.voice_styles[key]
        logger.info(f"Cleared {len(keys_to_remove)} voice styles for model {model_id}")

    def _handle_model_unloaded(self, model_id: str) -> None:
        """Override to add voice style cleanup."""
        self.on_model_unloaded(model_id)
        super()._handle_model_unloaded(model_id)
