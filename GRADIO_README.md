# Gradio Interface for EPUB to Audiobook Converter

This project provides a user-friendly web interface for converting EPUB files to audiobooks using various text-to-speech (TTS) engines.

## Features

- **Web-based Interface**: No command line knowledge required
- **Multiple TTS Engines**: Support for TTS, XTTS, OpenAI, Edge, Kokoro, and Kyutai
- **Real-time Progress**: Live conversion progress and logs
- **File Upload**: Direct EPUB/TXT file upload
- **Flexible Configuration**: Adjustable parameters for each engine
- **Multiple Output Formats**: M4B, WAV, and FLAC support
- **Chapter Selection**: Choose specific chapters to convert
- **Advanced Options**: Debug mode, skip links/footnotes, etc.

## Installation

### Prerequisites
- Python 3.7+
- FFmpeg (for audio processing)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

## Usage

### Quick Start

1. **Launch the interface:**
   ```bash
   python3 run_gradio.py
   ```
   Or use the shell script:
   ```bash
   ./run_gradio.sh
   ```

2. **Open your browser** to [http://localhost:7860](http://localhost:7860)

3. **Upload your EPUB file** using the file upload button

4. **Select your TTS engine** from the dropdown

5. **Choose a speaker voice** (options vary by engine)

6. **Configure settings** (start/end chapters, output format, etc.)

7. **Click "Convert to Audiobook"** and wait for completion

### TTS Engines

#### 1. TTS (Coqui TTS)
- **Default engine** - uses pre-trained models
- **Speakers**: p335, p307, p225, etc.
- **No additional setup required**

#### 2. XTTS
- **High-quality voice cloning**
- **Voice samples**: Upload your own voice samples
- **Format**: WAV/MP3 files separated by commas

#### 3. OpenAI TTS
- **Cloud-based** high-quality voices
- **API Key required**: Get from [OpenAI Platform](https://platform.openai.com)
- **Speakers**: alloy, echo, fable, onyx, nova, shimmer

#### 4. Edge TTS
- **Microsoft Edge voices**
- **No API key required**
- **Speakers**: en-US-AriaNeural, en-US-JennyNeural, etc.

#### 5. Kokoro
- **Fast, lightweight TTS**
- **Reading speed adjustable**
- **Speakers**: af_sky, af_bella, am_adam, etc.

#### 6. Kyutai
- **Latest neural TTS**
- **Multiple voice options**
- **Speakers**: af, am, bf, bm, etc.

### Configuration Options

#### Basic Settings
- **Start Chapter**: First chapter to convert (1 = beginning)
- **End Chapter**: Last chapter to convert (999 = all chapters)
- **Threads**: Number of parallel processing threads
- **Output Format**: m4b, wav, or flac
- **Bitrate**: Audio quality (69k, 128k, 192k)

#### Advanced Options
- **Minimum Match Ratio**: Whisper verification (0 = disabled)
- **Debug Mode**: Enable detailed logging
- **Skip Links**: Remove HTML links from text
- **Skip Footnotes**: Remove footnotes from text
- **Say Part Numbers**: Announce chapter numbers
- **Disable DeepSpeed**: Turn off DeepSpeed optimization
- **Skip Text Cleanup**: Skip text preprocessing

## Examples

### Example 1: Basic EPUB Conversion
1. Upload `mybook.epub`
2. Select engine: `tts`
3. Select speaker: `p335`
4. Click convert

### Example 2: OpenAI TTS
1. Upload `mybook.epub`
2. Select engine: `openai`
3. Enter API key
4. Select speaker: `onyx`
5. Click convert

### Example 3: XTTS with Custom Voice
1. Upload `mybook.epub`
2. Select engine: `xtts`
3. Enter voice samples: `voice1.wav,voice2.wav`
4. Click convert

## Troubleshooting

### Common Issues

**"FFmpeg not found"**
- Install FFmpeg using your system's package manager

**"CUDA out of memory"**
- Reduce threads to 1
- Use CPU instead of GPU
- Try smaller models

**"OpenAI API key required"**
- Sign up at [OpenAI Platform](https://platform.openai.com)
- Generate an API key
- Enter it in the interface

**"Module not found"**
- Run `pip install -r requirements.txt`
- Ensure all dependencies are installed

### Performance Tips

- **GPU**: Use CUDA for faster processing
- **Threads**: Increase for CPU processing
- **Memory**: Monitor RAM usage for large books
- **Storage**: Ensure sufficient disk space

## Development

### Adding New TTS Engines

1. Create a new class in `epub2tts.py`
2. Add engine to `ENGINE_SPEAKERS` dictionary
3. Update argument parser if needed
4. Test with sample files

### Customization

- **Themes**: Modify the Gradio theme in `create_interface()`
- **Layout**: Adjust column widths and component placement
- **Validation**: Add custom file validation rules
- **Logging**: Enhance output streaming

## API Usage

### Command Line Alternative
```bash
python3 epub2tts.py mybook.epub --engine tts --speaker p335
```

### Programmatic Usage
```python
from epub2tts import EpubToAudiobook

mybook = EpubToAudiobook(
    source="mybook.epub",
    engine="tts",
    speaker="p335"
)
mybook.get_chapters_epub(speaker="p335")
mybook.read_book(engine="tts", speaker="p335")
```

## Support

- **Issues**: Report on GitHub
- **Documentation**: Check README files
- **Community**: Join discussions
- **Updates**: Pull latest changes

## License

This project is licensed under the MIT License - see the LICENSE file for details.
