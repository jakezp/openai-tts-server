#!/bin/bash

# Test long-form text generation with StyleTTS2

LONG_TEXT="Hello, this is a test of the long-form text generation capability. The system should automatically split this text into manageable chunks and synthesize each one separately. This is important because the BERT model has a maximum sequence length of 512 tokens. When text exceeds this limit, we need to intelligently split it into sentences or smaller chunks. The splitting should be done in a way that preserves natural sentence boundaries, so the speech sounds natural and coherent. Each chunk is synthesized independently, and then the audio segments are concatenated together with small pauses in between. This approach allows us to handle arbitrarily long text inputs while maintaining high quality speech synthesis. The user should not notice any difference in quality compared to shorter texts, except for the natural pauses between chunks."

echo "Testing long-form synthesis..."
echo "Text length: ${#LONG_TEXT} characters"
echo ""

curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"jakezp/StyleTTS2-LibriTTS\", \"input\": \"$LONG_TEXT\", \"voice\": \"female_lisa\"}" \
  --output test_long_form.wav \
  --write-out "\nHTTP %{http_code} - Size: %{size_download} bytes\n" \
  -w "\nTime: %{time_total}s\n"

echo ""
echo "Audio saved to: test_long_form.wav"
