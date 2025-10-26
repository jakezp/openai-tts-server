# Speaches - OpenAI-Compatible TTS/STT Server

`speaches` is an OpenAI API-compatible server supporting streaming transcription, translation, and speech generation. This project aims to be **Ollama for TTS/STT models** - making it easy to run multiple speech models with a unified API.

## Powered By

- **Speech-to-Text**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **Text-to-Speech**: 
  - [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) - #1 in TTS Arena
  - [StyleTTS2](https://github.com/yl4579/StyleTTS2) - High-quality neural TTS with voice cloning
  - [Piper](https://github.com/rhasspy/piper) - Fast, lightweight TTS

For full documentation, visit: [speaches.ai](https://speaches.ai/)

## Features

### Core Capabilities

- **OpenAI API Compatible** - All tools and SDKs that work with OpenAI's API work with Speaches
- **Streaming Support** - Real-time transcription via SSE (no waiting for full audio processing)
- **Dynamic Model Management** - Automatic model loading/unloading based on usage
- **GPU & CPU Support** - Run on your preferred hardware
- **Docker Ready** - Easy deployment via [Docker Compose](https://speaches.ai/installation/)
- **Highly Configurable** - Extensive [configuration options](https://speaches.ai/configuration/)

### Audio Generation ([Realtime API](https://speaches.ai/usage/realtime-api))

- Text → Audio: Generate spoken summaries
- Audio → Text: Transcription and sentiment analysis
- Audio → Audio: Speech-to-speech model interactions

### Text-to-Speech Engines

- **Kokoro** - Ranked #1 in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)
- **StyleTTS2** - 24kHz high-quality synthesis, voice cloning, 18+ languages, 14 preset voices
- **Piper** - Fast and lightweight TTS

## Demos

### Realtime API

https://github.com/user-attachments/assets/457a736d-4c29-4b43-984b-05cc4d9995bc

(Excuse the breathing lol. Didn't have enough time to record a better demo)

### Streaming Transcription

TODO

### Speech Generation

https://github.com/user-attachments/assets/0021acd9-f480-4bc3-904d-831f54c4d45b

## Quick Start

### Text-to-Speech with StyleTTS2

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jakezp/StyleTTS2-LibriTTS",
    "input": "Hello world! This is high-quality neural speech synthesis.",
    "voice": "female_anna"
  }' \
  --output speech.wav
```

**Available Voices**: `female_anna`, `male_gavin`, `female_emily`, `male_thomas`, and 10 more preset voices.

**Voice Cloning**: Use `"voice": "file:///path/to/reference.wav"` to clone any voice.

**Custom Parameters**: Adjust `alpha` (0-1), `beta` (0-1), and `diffusion_steps` (1-20) for fine control.

## Attribution

This project builds upon and integrates several excellent open-source projects:

- **[Speaches](https://github.com/speaches-ai/speaches)** - The foundation of this OpenAI-compatible TTS/STT server (MIT License)
- **[StyleTTS2](https://github.com/yl4579/StyleTTS2)** - High-quality neural text-to-speech synthesis with style modeling (MIT License)
- **[StyleTTS2 HuggingFace Space](https://huggingface.co/spaces/Pendrokar/style-tts-2)** - Reference implementation and inspiration for integration
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** - Fast speech-to-text transcription
- **[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)** - State-of-the-art TTS model
- **[Piper](https://github.com/rhasspy/piper)** - Lightweight TTS engine

### License

This project is a combination and integration of the above works. All original projects retain their respective licenses (primarily MIT License). This derivative work is provided as-is for research and development purposes.

**We express our gratitude to all the original authors and contributors of these projects.** Their excellent work makes this integration possible.

## Support & Issues

Please create an issue if you:
- Find a bug
- Have a question
- Want to suggest a feature

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

**Note**: This is a community integration project combining multiple open-source TTS/STT engines into a unified OpenAI-compatible API. It is not affiliated with OpenAI.
