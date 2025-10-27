"""
StyleTTS2 inference engine.
Adapted from styletts2importable.py to work within the Speaches architecture.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import huggingface_hub
import librosa
import nltk
import numpy as np
import phonemizer
import torch
import torchaudio
import yaml
from munch import Munch
from nltk.tokenize import word_tokenize

if TYPE_CHECKING:
    from torch import nn

from speaches.executors.styletts2.models import build_model, load_ASR_models, load_F0_models
from speaches.executors.styletts2.modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
from speaches.executors.styletts2.text_utils import TextCleaner
from speaches.executors.styletts2.utils import recursive_munch  # Import from utils.py file
from speaches.executors.styletts2.utils.plbert.util import load_plbert

logger = logging.getLogger(__name__)

# Download NLTK data on first import
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

# Language name mappings for NLTK tokenization
LANG_NAMES = {
    "en-us": "english",
    "cs": "czech",
    "da": "danish",
    "nl": "dutch",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "el": "greek",
    "it": "italian",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "sl": "slovene",
    "es": "spanish",
    "sv": "swedish",
    "tr": "turkish",
}

# Language code normalization mapping for espeak compatibility
# Maps common variants to espeak-supported codes
ESPEAK_LANG_MAP = {
    # English variants - espeak supports these natively
    "en": "en-us",
    "en-us": "en-us",
    "en_us": "en-us",
    "en-gb": "en-gb",
    "en_gb": "en-gb",
    "en-029": "en-029",  # Caribbean
    "en-scotland": "en-gb-scotland",
    "en-gb-scotland": "en-gb-scotland",
    # Other languages
    "cs": "cs",
    "da": "da",
    "nl": "nl",
    "et": "et",
    "fi": "fi",
    "fr": "fr-fr",
    "fr-fr": "fr-fr",
    "fr_fr": "fr-fr",
    "fr-be": "fr-be",
    "fr_be": "fr-be",
    "fr-ch": "fr-ch",
    "fr_ch": "fr-ch",
    "de": "de",
    "el": "el",
    "it": "it",
    "no": "nb",
    "nb": "nb",
    "pl": "pl",
    "pt": "pt",
    "pt-br": "pt-br",
    "pt_br": "pt-br",
    "ru": "ru",
    "sl": "sl",
    "es": "es",
    "es-419": "es-419",
    "es_419": "es-419",
    "sv": "sv",
    "tr": "tr",
}


def normalize_language_code(lang: str) -> str:
    """
    Normalize language codes to espeak-compatible format.
    
    Handles variants like:
    - en_us, en-us -> en-us
    - en_gb, en-gb -> en-gb (preserves British English)
    - fr_fr, fr-fr -> fr-fr
    - pt_br, pt-br -> pt-br
    - etc.
    
    Args:
        lang: Language code (e.g., "en-us", "en_US", "en-gb", "fr-fr")
        
    Returns:
        Normalized espeak-compatible language code
    """
    # Convert to lowercase and replace underscores with hyphens
    lang = lang.lower().replace("_", "-")
    
    # Check if it's already in our mapping
    if lang in ESPEAK_LANG_MAP:
        return ESPEAK_LANG_MAP[lang]
    
    # Try extracting just the base language code (e.g., "en" from "en-au")
    # Only fallback to base if we don't have a specific variant
    if "-" in lang:
        base_lang = lang.split("-")[0]
        if base_lang in ESPEAK_LANG_MAP:
            return ESPEAK_LANG_MAP[base_lang]
    
    # Default to en-us if completely unknown
    return "en-us"


SAMPLE_RATE = 24000


def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Convert lengths to mask tensor."""
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


class StyleTTS2:
    """StyleTTS2 model wrapper for text-to-speech synthesis."""

    def __init__(
        self,
        model_id: str = "yl4579/StyleTTS2-LibriTTS",
        device: Literal["cpu", "cuda", "auto"] = "auto",
    ) -> None:
        """
        Initialize StyleTTS2 model.

        Args:
            model_id: HuggingFace model repository ID
            device: Device to run inference on
        """
        self.model_id = model_id
        self.device = self._get_device(device)
        logger.info(f"Initializing StyleTTS2 on device: {self.device}")

        # Initialize mel spectrogram transform
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        )
        self.mel_mean = -4
        self.mel_std = 4

        # Initialize text cleaner
        self.text_cleaner = TextCleaner()

        # Initialize phonemizer (default to English)
        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )

        # Load model components
        self._load_model()

    def _get_device(self, device: str) -> str:
        """Determine the device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            # MPS support disabled for now due to compatibility issues
            # elif torch.backends.mps.is_available():
            #     return "mps"
            return "cpu"
        return device

    def _load_model(self) -> None:
        """Load all model components from HuggingFace."""
        logger.info(f"Loading StyleTTS2 model: {self.model_id}")

        # Download config file
        config_path = huggingface_hub.hf_hub_download(
            repo_id=self.model_id,
            filename="Models/LibriTTS/config.yml",
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Download utility models from GitHub repository
        def _download_github_file(github_url: str, local_path: Path) -> str:
            """Download a file from GitHub repository."""
            import urllib.request
            
            # Convert GitHub tree URL to raw file URL
            raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/tree/main/", "/main/")
            
            # Create directory if it doesn't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading {raw_url} -> {local_path}")
            urllib.request.urlretrieve(raw_url, local_path)
            return str(local_path)

        def _get_utility_model_path(hf_path: str) -> str | None:
            """Get utility model path by downloading from GitHub if needed."""
            if not hf_path:
                return None
            
            if hf_path.startswith("Utils/"):
                current_dir = Path(__file__).parent  
                relative_path = hf_path[6:]  # Remove "Utils/" prefix
                
                # Handle directories (like PLBERT/)
                if hf_path.endswith("/"):
                    return _get_utility_directory(hf_path, relative_path.rstrip("/").lower())
                
                # Handle files
                cache_path = Path.home() / ".cache" / "styletts2" / relative_path.lower()
                
                # If cached and not an LFS pointer, use it
                if cache_path.exists() and cache_path.stat().st_size > 1000:
                    logger.info(f"Using cached utility model: {cache_path}")
                    return str(cache_path)
                
                # Download from GitHub
                try:
                    github_base = "https://github.com/yl4579/StyleTTS2/tree/main"
                    github_url = f"{github_base}/{hf_path}"
                    
                    # Convert to raw URL for download
                    raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/tree/main/", "/main/")
                    
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Downloading utility model from GitHub: {raw_url}")
                    
                    import urllib.request
                    urllib.request.urlretrieve(raw_url, cache_path)
                    
                    logger.info(f"Downloaded utility model: {hf_path} -> {cache_path}")
                    return str(cache_path)
                    
                except Exception as e:
                    logger.error(f"Failed to download {hf_path} from GitHub: {e}")
                    
                    # Try local fallback for development
                    fallback_path = current_dir / "utils" / relative_path.lower()
                    if fallback_path.exists() and fallback_path.stat().st_size > 1000:
                        logger.warning(f"Using local fallback: {fallback_path}")
                        return str(fallback_path)
                    else:
                        logger.error(f"Utility model not available: {hf_path}")
                        return None
            
            return hf_path

        def _get_utility_directory(full_path: str, dir_name: str) -> str | None:
            """Download and cache an entire utility directory from GitHub."""
            cache_dir = Path.home() / ".cache" / "styletts2" / dir_name
            
            # Define required files for each directory
            if dir_name == "plbert":
                required_files = [
                    'config.yml',
                    'step_1000000.t7', 
                    'util.py'
                ]
            else:
                logger.error(f"Unknown utility directory: {dir_name}")
                return None
            
            # Check if all files exist in cache
            all_cached = all((cache_dir / filename).exists() for filename in required_files)
            
            if all_cached:
                logger.info(f"Using cached directory: {cache_dir}")
                return str(cache_dir)
            
            # Download missing files
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            for filename in required_files:
                file_path = cache_dir / filename
                if not file_path.exists():
                    try:
                        url = f"https://raw.githubusercontent.com/yl4579/StyleTTS2/main/{full_path}{filename}"
                        logger.info(f"Downloading {filename} from GitHub...")
                        
                        import urllib.request
                        urllib.request.urlretrieve(url, file_path)
                        logger.info(f"Downloaded: {file_path}")
                        
                    except Exception as e:
                        logger.error(f"Failed to download {filename}: {e}")
                        # Try fallback to local file
                        current_dir = Path(__file__).parent
                        local_file = current_dir / "utils" / dir_name / filename
                        if local_file.exists():
                            import shutil
                            shutil.copy2(local_file, file_path)
                            logger.info(f"Copied from local: {file_path}")
                        else:
                            logger.error(f"Cannot download or find local file: {filename}")
                            return None
            
            logger.info(f"Directory ready: {cache_dir}")
            return str(cache_dir)

        # Load utility models (download from GitHub if needed)
        asr_config_path = _get_utility_model_path(config.get("ASR_config", False))
        asr_path = _get_utility_model_path(config.get("ASR_path", False))
        self.text_aligner = load_ASR_models(asr_path, asr_config_path)

        # Load pretrained F0 model  
        f0_path = _get_utility_model_path(config.get("F0_path", False))
        self.pitch_extractor = load_F0_models(f0_path)

        # Load BERT model
        bert_path = _get_utility_model_path(config.get("PLBERT_dir", False))
        self.plbert = load_plbert(bert_path)

        # Build model
        model_params = recursive_munch(config["model_params"])
        self.model_params = model_params
        self.model = build_model(model_params, self.text_aligner, self.pitch_extractor, self.plbert)

        # Set to eval mode and move to device
        for key in self.model:
            self.model[key].eval()
            self.model[key].to(self.device)

        # Load model weights
        weights_path = huggingface_hub.hf_hub_download(
            repo_id=self.model_id,
            filename="Models/LibriTTS/epochs_2nd_00020.pth",
        )
        params_whole = torch.load(weights_path, map_location="cpu", weights_only=False)
        params = params_whole["net"]

        # Load state dicts
        for key in self.model:
            if key in params:
                logger.debug(f"Loading {key}")
                try:
                    self.model[key].load_state_dict(params[key])
                except RuntimeError:
                    # Handle DataParallel models
                    from collections import OrderedDict

                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] if k.startswith("module.") else k  # remove `module.` prefix
                        new_state_dict[name] = v
                    self.model[key].load_state_dict(new_state_dict, strict=False)

        # Set to eval mode again
        for key in self.model:
            self.model[key].eval()

        # Initialize diffusion sampler
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )

        logger.info("StyleTTS2 model loaded successfully")

    def preprocess_audio(self, wave: np.ndarray) -> torch.Tensor:
        """Convert audio waveform to mel spectrogram."""
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mel_mean) / self.mel_std
        return mel_tensor

    def compute_style(self, audio_path: str | Path) -> torch.Tensor:
        """
        Compute style embedding from reference audio.

        Args:
            audio_path: Path to reference audio file

        Returns:
            Style embedding tensor
        """
        # Convert Path to string for librosa compatibility
        audio_path_str = str(audio_path) if isinstance(audio_path, Path) else audio_path
        wave, sr = librosa.load(audio_path_str, sr=SAMPLE_RATE)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        mel_tensor = self.preprocess_audio(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    def synthesize(
        self,
        text: str,
        ref_style: torch.Tensor,
        lang: str = "en-us",
        alpha: float = 0.3,
        beta: float = 0.7,
        diffusion_steps: int = 5,
        embedding_scale: float = 1.0,
    ) -> tuple[np.ndarray, str]:
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize
            ref_style: Reference style embedding
            lang: Language code
            alpha: Style blending factor
            beta: Style mixing factor
            diffusion_steps: Number of diffusion steps (3-15)
            embedding_scale: Embedding scale factor

        Returns:
            Tuple of (audio waveform, IPA phonemes)
        """
        text = text.strip()

        # Normalize language code for espeak compatibility
        normalized_lang = normalize_language_code(lang)

        # Handle IPA phonemes within brackets []
        ipa_pattern = r"\[[^\]]*\]"
        text = text.replace("[]", "")
        ipa_sections = re.findall(ipa_pattern, text)

        # Replace IPA sections with placeholder
        if ipa_sections:
            text = re.sub(ipa_pattern, "[]", text, 0, re.MULTILINE)

        # Phonemize text based on language
        if normalized_lang in LANG_NAMES or lang in LANG_NAMES:
            # Use original lang for LANG_NAMES lookup, normalized_lang for espeak
            lang_name = LANG_NAMES.get(lang, LANG_NAMES.get(normalized_lang, "english"))
            local_phonemizer = phonemizer.backend.EspeakBackend(
                language=normalized_lang, preserve_punctuation=True, with_stress=True
            )
            ps = local_phonemizer.phonemize([text])
            ps = word_tokenize(ps[0], language=lang_name)
            ps = " ".join(ps)
        elif lang == "jb":
            # Lojban language support
            from speaches.executors.styletts2 import lojban

            ps = lojban.lojban2ipa(text, "vits")
        else:
            ps = text

        # Restore IPA sections
        if ipa_sections:
            for ipa in ipa_sections:
                ps = ps.replace("[ ]", ipa, 1)

        # Tokenize
        tokens = self.text_cleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            # Encode text
            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            # Style diffusion
            s_pred = self.sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_style,
                num_steps=diffusion_steps,
            ).squeeze(1)

            # Mix styles
            s = s_pred[:, 128:]
            ref = s_pred[:, :128]
            ref = alpha * ref + (1 - alpha) * ref_style[:, :128]
            s = beta * s + (1 - beta) * ref_style[:, 128:]

            # Predict duration
            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            # Create alignment
            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # Encode prosody
            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            # Predict F0 and energy
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            # Encode alignment
            asr = t_en @ pred_aln_trg.unsqueeze(0).to(self.device)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            # Decode to audio
            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        # Remove artifact at the end
        audio = out.squeeze().cpu().numpy()[..., :-50]
        return audio, ps
