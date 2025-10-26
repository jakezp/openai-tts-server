"""StyleTTS2 model registry for HuggingFace model discovery."""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Literal

import huggingface_hub
from pydantic import BaseModel, computed_field

from speaches.api_types import Model
from speaches.executors.styletts2.inference import SAMPLE_RATE
from speaches.hf_utils import (
    HfModelFilter,
    extract_language_list,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
)
from speaches.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

LIBRARY_NAME = "pytorch"
TASK_NAME_TAG = "text-to-speech"
TAGS = {"speaches", "styletts2"}


class StyleTTS2ModelFiles(BaseModel):
    """File paths for StyleTTS2 model components."""

    config: Path
    weights: Path
    voices_dir: Path | None = None


class StyleTTS2ModelVoice(BaseModel):
    """Voice configuration for StyleTTS2."""

    name: str
    language: str
    gender: Literal["male", "female"] | None = None

    @computed_field
    @property
    def id(self) -> str:
        return self.name


# Preset voices for StyleTTS2-LibriTTS model
PRESET_VOICES = [
    StyleTTS2ModelVoice(name="female_anna", language="en-us", gender="female"),
    StyleTTS2ModelVoice(name="female_default", language="en-us", gender="female"),
    StyleTTS2ModelVoice(name="female_emily", language="en-us", gender="female"),
    StyleTTS2ModelVoice(name="female_lisa", language="en-us", gender="female"),
    StyleTTS2ModelVoice(name="female_sarah", language="en-us", gender="female"),
    StyleTTS2ModelVoice(name="male_gavin", language="en-us", gender="male"),
    StyleTTS2ModelVoice(name="male_jakez", language="en-us", gender="male"),
    StyleTTS2ModelVoice(name="male_john", language="en-us", gender="male"),
    StyleTTS2ModelVoice(name="male_nima", language="en-us", gender="male"),
    StyleTTS2ModelVoice(name="male_old", language="en-us", gender="male"),
    StyleTTS2ModelVoice(name="male_robert", language="en-us", gender="male"),
    StyleTTS2ModelVoice(name="male_thomas", language="en-us", gender="male"),
    StyleTTS2ModelVoice(name="male_vinay", language="en-us", gender="male"),
    StyleTTS2ModelVoice(name="male_yinghao", language="en-us", gender="male"),
]


class StyleTTS2Model(Model):
    """Model metadata for StyleTTS2."""

    sample_rate: int
    voices: list[StyleTTS2ModelVoice]
    supports_voice_cloning: bool = True
    supports_multi_language: bool = True


hf_model_filter = HfModelFilter(
    library_name=LIBRARY_NAME,
    task=TASK_NAME_TAG,
    tags={"styletts2"},  # Only require styletts2 tag, not speaches
)


class StyleTTS2ModelRegistry(ModelRegistry):
    """Registry for discovering StyleTTS2 models."""

    def list_remote_models(self) -> Generator[StyleTTS2Model, None, None]:
        """List available StyleTTS2 models from HuggingFace."""
        # For now, we'll just yield the main StyleTTS2 model
        # In the future, this could be expanded to discover more models
        yield StyleTTS2Model(
            id="jakezp/StyleTTS2-LibriTTS",
            created=0,  # Unknown creation date
            owned_by="yl4579",
            language=["en", "multi"],  # Supports multiple languages
            task=TASK_NAME_TAG,
            sample_rate=SAMPLE_RATE,
            voices=PRESET_VOICES,
            supports_voice_cloning=True,
            supports_multi_language=True,
        )

    def list_local_models(self) -> Generator[StyleTTS2Model, None, None]:
        """List locally cached StyleTTS2 models."""
        cached_model_repos_info = get_cached_model_repos_info()
        for cached_repo_info in cached_model_repos_info:
            # Check if this is a StyleTTS2 model
            if "StyleTTS2" in cached_repo_info.repo_id:
                model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
                if model_card_data is None:
                    continue

                # Extract languages if available
                languages = extract_language_list(model_card_data)
                if not languages:
                    languages = ["en", "multi"]

                yield StyleTTS2Model(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=languages,
                    task=TASK_NAME_TAG,
                    sample_rate=SAMPLE_RATE,
                    voices=PRESET_VOICES,
                    supports_voice_cloning=True,
                    supports_multi_language=True,
                )


# Create singleton instance
styletts2_model_registry = StyleTTS2ModelRegistry(hf_model_filter=hf_model_filter)
