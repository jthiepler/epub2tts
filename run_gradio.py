#!/usr/bin/env python3
"""
Simple launcher for the Gradio interface
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gradio_interface import create_interface

if __name__ == "__main__":
    print("Starting EPUB to Audiobook Converter...")
    print("Access the interface at http://localhost:7860")
    
    app = create_interface()
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )
