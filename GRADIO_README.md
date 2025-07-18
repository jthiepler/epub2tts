# Gradio Interface for epub2tts

This directory now includes a web-based interface for converting EPUB files to audiobooks using Gradio.

## Features

- **Web-based UI**: No command line required
- **Real-time streaming**: See conversion progress as it happens
- **Multiple TTS engines**: Support for tts, xtts, openai, edge, and kokoro
- **Dynamic speaker selection**: Speakers change based on selected engine
- **File upload**: Direct EPUB/TXT file upload
- **Configurable options**: All CLI options available through the interface

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Interface
```bash
# Method 1: Using the launcher script
python3 run_gradio.py

# Method 2: Using the shell script
chmod +x run_gradio.sh
./run_gradio.sh
```

### 3. Access the Interface
Open your browser to: http://localhost:7860

## Usage

1. **Upload your EPUB or TXT file** using the file upload button
2. **Select your TTS engine** from the dropdown
3. **Choose a speaker voice** (options change based on engine)
4. **Configure settings**:
   - Start/end chapters
   - Output format (m4b, wav, flac)
   - Threads for processing
   - Advanced options
5. **Click "Convert to Audiobook"** and watch the real-time progress

## Engine-Specific Setup

### OpenAI TTS
- Requires OpenAI API key
- Voices: alloy, echo, fable, onyx, nova, shimmer

### XTTS
- Optional: Provide voice sample files for custom voices
- Default speakers available

### Edge TTS
- Uses Microsoft Edge voices
- No additional setup required

### Kokoro
- Uses Kokoro TTS
- Reading speed adjustable

### Coqui TTS
- Uses default Coqui TTS models
- Multiple speaker voices available

## Troubleshooting

### Common Issues

1. **"Module not found" errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Port already in use**
   - Change port in run_gradio.py: `server_port=7861`

3. **Permission errors**
   ```bash
   chmod +x run_gradio.py run_gradio.sh
   ```

4. **CUDA/GPU issues**
   - Use CPU mode by setting threads=1
   - Disable DeepSpeed if needed

## Development

The interface consists of:
- `gradio_interface.py`: Main Gradio application
- `run_gradio.py`: Simple launcher script
- `run_gradio.sh`: Shell script launcher

## Advanced Usage

### Custom Configuration
Edit `gradio_interface.py` to:
- Add new TTS engines
- Modify speaker lists
- Change default settings
- Customize UI appearance

### Docker Support
The interface can be run in Docker containers using the provided Dockerfiles.
