#!/bin/bash
# Launch script for Gradio interface

echo "Starting EPUB to Audiobook Converter (Gradio Interface)..."
echo "Make sure you have installed the required dependencies:"
echo "pip install -r requirements.txt"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if gradio is installed
python3 -c "import gradio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Gradio is not installed. Run: pip install gradio"
    exit 1
fi

# Launch the interface
python3 run_gradio.py
