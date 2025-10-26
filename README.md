# Speaches

`speaches` is an OpenAI API-compatible server supporting streaming transcription, translation, and speech generation. Speech-to-Text is powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and for Text-to-Speech [piper](https://github.com/rhasspy/piper), [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M), and [StyleTTS2](https://github.com/yl4579/StyleTTS2) are used. This project aims to be Ollama, but for TTS/STT models.

See the documentation for installation instructions and usage: [speaches.ai](https://speaches.ai/)

## Features:

- OpenAI API compatible. All tools and SDKs that work with OpenAI's API should work with `speaches`.
- Audio generation (chat completions endpoint) | [OpenAI Documentation](https://platform.openai.com/docs/guides/realtime)
  - Generate a spoken audio summary of a body of text (text in, audio out)
  - Perform sentiment analysis on a recording (audio in, text out)
  - Async speech to speech interactions with a model (audio in, audio out)
- Streaming support (transcription is sent via SSE as the audio is transcribed. You don't need to wait for the audio to fully be transcribed before receiving it).
- Dynamic model loading / offloading. Just specify which model you want to use in the request and it will be loaded automatically. It will then be unloaded after a period of inactivity.
- Text-to-Speech via:
  - **Kokoro** - Ranked #1 in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)
  - **StyleTTS2** - High-quality neural TTS with voice cloning support (24kHz, 18+ languages, 14 preset voices)
  - **Piper** - Fast, lightweight TTS
- GPU and CPU support.
- [Deployable via Docker Compose / Docker](https://speaches.ai/installation/)
- [Realtime API](https://speaches.ai/usage/realtime-api)
- [Highly configurable](https://speaches.ai/configuration/)

Please create an issue if you find a bug, have a question, or a feature suggestion.

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
