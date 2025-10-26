"""StyleTTS2 Gradio UI Tab"""

from pathlib import Path
from tempfile import NamedTemporaryFile

import gradio as gr  # type: ignore
import numpy as np  # type: ignore
from speaches.config import Config
from speaches.executors.styletts2.registry import PRESET_VOICES
from speaches.executors.styletts2.model_manager import StyleTTS2ModelManager
from speaches.executors.shared.registry import ExecutorRegistry

# Language options for StyleTTS2
LANGUAGE_OPTIONS = [
    ['English (US)', 'en-us'],
    # Artificial language
    ['Lojban', 'jb'],
    # Natural non-native languages
    ['Czech (Non-native)', 'cs'],
    ['Danish (Non-native)', 'da'],
    ['Dutch (Non-native)', 'nl'],
    ['Estonian (Non-native)', 'et'],
    ['Finnish (Non-native)', 'fi'],
    ['French (Non-native)', 'fr'],
    ['German (Non-native)', 'de'],
    ['Greek (Non-native)', 'el'],
    ['Italian (Non-native)', 'it'],
    ['Norwegian (Non-native)', 'no'],
    ['Polish (Non-native)', 'pl'],
    ['Portuguese (Non-native)', 'pt'],
    ['Russian (Non-native)', 'ru'],
    ['Slovene (Non-native)', 'sl'],
    ['Spanish (Non-native)', 'es'],
    ['Swedish (Non-native)', 'sv'],
    ['Turkish (Non-native)', 'tr'],
]

DEFAULT_MODEL_ID = "yl4579/StyleTTS2-LibriTTS"


def create_styletts2_tab(config: Config) -> None:
    """Create the StyleTTS2 Gradio interface tab."""

    # Get the executor registry
    executor_registry = ExecutorRegistry(config)
    
    # Multi-Voice tab synthesis
    async def multi_voice_synthesize(
        text: str,
        voice: str,
        lang: str,
        diffusion_steps: int,
    ) -> tuple[int, np.ndarray]:
        """Synthesize speech using a preset voice."""
        if not text.strip():
            raise gr.Error("You must enter some text")
        if len(text) > 50000:
            raise gr.Error("Text must be <50k characters")
        
        # Get the StyleTTS2 model manager
        model_manager = executor_registry.executor_dict["styletts2"].model_manager  # type: ignore
        assert isinstance(model_manager, StyleTTS2ModelManager)
        
        # Load the model
        model_wrapper = model_manager.load_model(DEFAULT_MODEL_ID)
        model = model_wrapper.model
        assert model is not None
        
        # Get voice style
        voice_style = model_manager.get_voice_style(model, voice)
        
        # Synthesize
        audio = model.synthesize(
            text=text,
            ref_style=voice_style,
            lang=lang,
            alpha=0.3,
            beta=0.7,
            diffusion_steps=diffusion_steps,
            embedding_scale=1.0,
        )
        
        return (24000, audio)
    
    # Voice cloning synthesis
    async def voice_clone_synthesize(
        text: str,
        voice_file: str,
        diffusion_steps: int,
        embedding_scale: float,
        alpha: float,
        beta: float,
    ) -> tuple[int, np.ndarray]:
        """Synthesize speech using a custom voice."""
        if not text.strip():
            raise gr.Error("You must enter some text")
        if len(text) > 50000:
            raise gr.Error("Text must be <50k characters")
        if not voice_file:
            raise gr.Error("You must provide a voice audio file")
        if embedding_scale > 1.3 and len(text) < 20:
            gr.Warning("WARNING: You entered short text with high embedding scale, you may get static!")
        
        # Get the StyleTTS2 model manager
        model_manager = executor_registry.executor_dict["styletts2"].model_manager  # type: ignore
        assert isinstance(model_manager, StyleTTS2ModelManager)
        
        # Load the model
        model_wrapper = model_manager.load_model(DEFAULT_MODEL_ID)
        model = model_wrapper.model
        assert model is not None
        
        # Compute custom voice style
        voice_style = model_manager.compute_custom_voice_style(model, voice_file)
        
        # Synthesize
        audio = model.synthesize(
            text=text,
            ref_style=voice_style,
            lang="en-us",  # Default to English for voice cloning
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
        )
        
        return (24000, audio)

    # LJSpeech synthesis (random voice)
    async def ljspeech_synthesize(
        text: str,
        diffusion_steps: int,
    ) -> tuple[int, np.ndarray]:
        """Synthesize speech with random voice (LJSpeech style)."""
        if not text.strip():
            raise gr.Error("You must enter some text")
        if len(text) > 150000:
            raise gr.Error("Text must be <150k characters")
        
        # Get the StyleTTS2 model manager
        model_manager = executor_registry.executor_dict["styletts2"].model_manager  # type: ignore
        assert isinstance(model_manager, StyleTTS2ModelManager)
        
        # Load the model
        model_wrapper = model_manager.load_model(DEFAULT_MODEL_ID)
        model = model_wrapper.model
        assert model is not None
        
        # Use a random style (m-us-1 as default)
        voice_style = model_manager.get_voice_style(model, "m-us-1")
        
        # Synthesize
        audio = model.synthesize(
            text=text,
            ref_style=voice_style,
            lang="en-us",
            alpha=0.3,
            beta=0.7,
            diffusion_steps=diffusion_steps,
            embedding_scale=1.0,
        )
        
        return (24000, audio)

    with gr.Tab(label="StyleTTS2"):
        gr.Markdown("""
# StyleTTS 2

[Paper](https://arxiv.org/abs/2306.07691) - [Samples](https://styletts2.github.io/) - [Code](https://github.com/yl4579/StyleTTS2)

High-quality neural text-to-speech with style control and voice cloning capabilities.

**NOTE:** StyleTTS 2 performs better on longer texts. Short phrases like "hi" will produce lower-quality results than longer sentences.

**NOTE:** This is an English-first model, but supports 18+ languages with non-native accent.
""")
        
        with gr.Tabs():
            # Multi-Voice Tab
            with gr.Tab("Multi-Voice"):
                with gr.Row():
                    with gr.Column(scale=1):
                        mv_text = gr.Textbox(
                            label="Text",
                            info="What would you like StyleTTS 2 to read? Works better on full sentences. Also accepts IPA within [] brackets.",
                            lines=5,
                            placeholder="Enter text here...",
                        )
                        mv_voice = gr.Dropdown(
                            choices=[v.name for v in PRESET_VOICES],
                            label="Voice",
                            info="Select a preset voice",
                            value="m-us-2",
                        )
                        mv_lang = gr.Dropdown(
                            choices=LANGUAGE_OPTIONS,
                            label="Language",
                            value="en-us",
                        )
                        mv_steps = gr.Slider(
                            minimum=3,
                            maximum=15,
                            value=5,
                            step=1,
                            label="Diffusion Steps",
                            info="Higher = better quality but slower. Try lower steps first for faster generation.",
                        )
                    with gr.Column(scale=1):
                        mv_btn = gr.Button("Synthesize", variant="primary")
                        mv_audio = gr.Audio(
                            label="Synthesized Audio",
                            interactive=False,
                            show_download_button=True,
                        )
                        mv_btn.click(
                            multi_voice_synthesize,
                            inputs=[mv_text, mv_voice, mv_lang, mv_steps],
                            outputs=[mv_audio],
                        )
            
            # Voice Cloning Tab
            with gr.Tab("Voice Cloning"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="Text",
                            info="What would you like StyleTTS 2 to read? Works better on full sentences.",
                            lines=5,
                            placeholder="Enter text here...",
                        )
                        vc_voice = gr.Audio(
                            label="Voice Sample",
                            info="Upload a voice sample (max 300 seconds)",
                            type="filepath",
                            max_length=300,
                        )
                        vc_steps = gr.Slider(
                            minimum=3,
                            maximum=20,
                            value=10,
                            step=1,
                            label="Diffusion Steps",
                            info="Higher = better quality but slower",
                        )
                        vc_embscale = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=1,
                            step=0.1,
                            label="Embedding Scale",
                            info="WARNING: High values with short text will cause static!",
                        )
                        vc_alpha = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.3,
                            step=0.1,
                            label="Alpha (Timbre control)",
                            info="Defaults to 0.3",
                        )
                        vc_beta = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.7,
                            step=0.1,
                            label="Beta (Prosody control)",
                            info="Defaults to 0.7",
                        )
                    with gr.Column(scale=1):
                        vc_btn = gr.Button("Synthesize", variant="primary")
                        vc_audio = gr.Audio(
                            label="Synthesized Audio",
                            interactive=False,
                            show_download_button=True,
                        )
                        vc_btn.click(
                            voice_clone_synthesize,
                            inputs=[vc_text, vc_voice, vc_steps, vc_embscale, vc_alpha, vc_beta],
                            outputs=[vc_audio],
                        )
            
            # LJSpeech Tab
            with gr.Tab("LJSpeech"):
                gr.Markdown("""
**LJSpeech-style generation** - Uses a default voice for quick generation.
""")
                with gr.Row():
                    with gr.Column(scale=1):
                        lj_text = gr.Textbox(
                            label="Text",
                            info="What would you like StyleTTS 2 to read? Works better on full sentences.",
                            lines=5,
                            placeholder="Enter text here...",
                        )
                        lj_steps = gr.Slider(
                            minimum=3,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Diffusion Steps",
                            info="Higher = better quality but slower",
                        )
                    with gr.Column(scale=1):
                        lj_btn = gr.Button("Synthesize", variant="primary")
                        lj_audio = gr.Audio(
                            label="Synthesized Audio",
                            interactive=False,
                            show_download_button=True,
                        )
                        lj_btn.click(
                            ljspeech_synthesize,
                            inputs=[lj_text, lj_steps],
                            outputs=[lj_audio],
                        )
