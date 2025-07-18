# Gradio Interface for EPUB to Audiobook Converter

A modern web-based interface for converting EPUB files to audiobooks using various TTS engines.

## Features

- **📚 Multiple TTS Engines**: Support for TTS, XTTS, OpenAI, Edge, Kokoro, and Kyutai
- **🎭 Dynamic Speaker Selection**: Automatic speaker/voice options based on selected engine
- **📁 Easy File Upload**: Drag-and-drop EPUB/TXT file upload
- **⚙️ Advanced Options**: Configure threads, bitrate, output format, and more
- **🔄 Real-time Progress**: Live conversion logs and progress updates
- **💾 CPU/GPU Toggle**: Force CPU mode for Kyutai TTS to avoid CUDA memory issues

## Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Launch the interface
python run_gradio.py
```

### Usage
1. Open your browser to `http://localhost:7860`
2. Upload your EPUB or TXT file
3. Select your preferred TTS engine
4. Choose a speaker/voice
5. Configure output settings
6. Click "Convert to Audiobook"

## TTS Engines

| Engine | Description | Speakers Available |
|--------|-------------|-------------------|
| **tts** | Coqui TTS (default) | 400+ voices like p335, p307 |
| **xtts** | XTTS v2 with voice cloning | 30+ preset voices |
| **openai** | OpenAI TTS API | alloy, echo, fable, onyx, nova, shimmer |
| **edge** | Microsoft Edge TTS | 30+ voices including Aria, Jenny, Guy |
| **kokoro** | Kokoro TTS | af_sky, af_bella, am_adam, etc. |
| **kyutai** | Kyutai TTS | af, am, bf, bm voices |

## Configuration Options

### Basic Settings
- **Start/End Chapter**: Process specific chapters or entire book
- **Threads**: Number of parallel processing threads (1-8)
- **Output Format**: m4b, wav, or flac
- **Bitrate**: Audio quality (69k, 128k, 192k)

### Advanced Options
- **Minimum Match Ratio**: Whisper verification (0-100, 0 to disable)
- **Debug Mode**: Enable detailed logging
- **Skip Links**: Remove HTML links from text
- **Skip Footnotes**: Remove footnotes from text
- **Say Part Numbers**: Include chapter numbers in audio
- **Disable DeepSpeed**: Turn off DeepSpeed optimization
- **Skip Text Cleanup**: Skip text preprocessing

## Engine-Specific Options

### OpenAI TTS
- Requires API key (sk-...)
- Uses cloud-based processing

### XTTS
- Supports custom voice samples
- Upload WAV files separated by commas

### Kyutai TTS
- **CPU Mode**: Toggle to force CPU processing when CUDA memory is low
- Uses local processing

## Troubleshooting

### CUDA Out of Memory
For Kyutai TTS, enable "Use CPU for Kyutai TTS" checkbox to avoid CUDA memory issues.

### Missing Dependencies
```bash
# Install missing packages
pip install gradio
pip install kyutai-tts  # for Kyutai support
```

### File Upload Issues
- Ensure files are valid EPUB or TXT format
- Check file permissions
- Verify file is not corrupted

## Examples

### Basic Conversion
```bash
# Convert entire book with default settings
python run_gradio.py
# Then upload EPUB and click convert
```

### Advanced Usage
```bash
# Convert specific chapters with custom settings
# Use the web interface to configure:
# - Start chapter: 5
# - End chapter: 10
# - Engine: kokoro
# - Speaker: af_sky
```

## Development

### Adding New Engines
1. Create a new TTS class inheriting from `Text2WaveFile`
2. Add engine to `ENGINE_SPEAKERS` dictionary
3. Update UI components in `gradio_interface.py`

### Custom Voices
For XTTS, prepare voice samples:
```bash
# Record 2-3 minute samples
arecord -f cd -t wav sample1.wav
arecord -f cd -t wav sample2.wav
```

## Command Line Alternative
For batch processing or automation, use the original CLI:
```bash
python epub2tts.py book.epub --engine tts --speaker p335
```

## Support
- GitHub Issues: [epub2tts](https://github.com/jthiepler/epub2tts/issues)
- Documentation: Check README.md for CLI usage
