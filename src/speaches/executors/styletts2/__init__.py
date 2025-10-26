"""StyleTTS2 text-to-speech executor."""

import logging
import time
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from speaches.audio import resample_audio
from speaches.executors.styletts2.inference import SAMPLE_RATE, StyleTTS2

if TYPE_CHECKING:
    from speaches.executors.styletts2.model_manager import StyleTTS2ModelManager

logger = logging.getLogger(__name__)


def generate_audio_sync(
    model_manager: "StyleTTS2ModelManager",
    model_id: str,
    text: str,
    voice: str,
    *,
    lang: str = "en-us",
    alpha: float = 0.3,
    beta: float = 0.7,
    diffusion_steps: int = 5,
    embedding_scale: float = 1.0,
    sample_rate: int | None = None,
) -> Generator[bytes, None, None]:
    """
    Synchronous version of generate_audio for non-async contexts.

    Args:
        model_manager: StyleTTS2 model manager
        model_id: HuggingFace model ID
        text: Text to synthesize
        voice: Voice name or path to reference audio
        lang: Language code
        alpha: Style blending factor
        beta: Style mixing factor
        diffusion_steps: Number of diffusion steps
        embedding_scale: Embedding scale factor
        sample_rate: Target sample rate (if different from model default)

    Yields:
        Audio data as bytes
    """
    if sample_rate is None:
        sample_rate = SAMPLE_RATE

    start = time.perf_counter()

    with model_manager.load_model(model_id) as styletts2_model:
        # Get or compute voice style
        if voice.startswith("file://") or Path(voice).exists():
            # Custom voice from file
            voice_path = voice.replace("file://", "") if voice.startswith("file://") else voice
            ref_style = model_manager.compute_custom_voice_style(styletts2_model, voice_path)
        else:
            # Preset voice
            ref_style = model_manager.get_voice_style(styletts2_model, voice)

        # Synthesize audio
        audio_data, ipa = styletts2_model.synthesize(
            text=text,
            ref_style=ref_style,
            lang=lang,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
        )

        logger.debug(f"Generated IPA: {ipa}")

        # Convert to int16 and yield as bytes
        normalized_audio_data = (audio_data * np.iinfo(np.int16).max).astype(np.int16)
        audio_bytes = normalized_audio_data.tobytes()

        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            audio_bytes = resample_audio(audio_bytes, SAMPLE_RATE, sample_rate)

        # Yield the complete audio
        yield audio_bytes

    logger.info(f"Generated audio for {len(text)} characters in {time.perf_counter() - start:.2f}s")


__all__ = ["StyleTTS2", "SAMPLE_RATE", "generate_audio_sync"]
