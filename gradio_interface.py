#!/usr/bin/env python3
"""
Gradio interface for epub2tts
Provides a web-based UI for converting EPUB files to audiobooks
"""

import gradio as gr
import os
import sys
import tempfile
import threading
import queue
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import argparse
from epub2tts import EpubToAudiobook

# Available engines and their speakers
ENGINE_SPEAKERS = {
    "tts": ["p335", "p307", "p225", "p226", "p227", "p228", "p229", "p230", "p231", "p232", "p233", "p234", "p236", "p237", "p238", "p239", "p240", "p241", "p243", "p244", "p245", "p246", "p247", "p248", "p249", "p250", "p251", "p252", "p253", "p254", "p255", "p256", "p257", "p258", "p259", "p260", "p261", "p262", "p263", "p264", "p265", "p266", "p267", "p268", "p269", "p270", "p271", "p272", "p273", "p274", "p275", "p276", "p277", "p278", "p279", "p280", "p281", "p282", "p283", "p284", "p285", "p286", "p287", "p288", "p289", "p290", "p291", "p292", "p293", "p294", "p295", "p296", "p297", "p298", "p299", "p300", "p301", "p302", "p303", "p304", "p305", "p306", "p308", "p309", "p310", "p311", "p312", "p313", "p314", "p315", "p316", "p317", "p318", "p319", "p320", "p321", "p322", "p323", "p324", "p325", "p326", "p327", "p328", "p329", "p330", "p331", "p332", "p333", "p334", "p336", "p337", "p338", "p339", "p340", "p341", "p342", "p343", "p344", "p345", "p346", "p347", "p348", "p349", "p350", "p351", "p352", "p353", "p354", "p355", "p356", "p357", "p358", "p359", "p360", "p361", "p362", "p363", "p364", "p365", "p366", "p367", "p368", "p369", "p370", "p371", "p372", "p373", "p374", "p375", "p376", "p377", "p378", "p379", "p380", "p381", "p382", "p383", "p384", "p385", "p386", "p387", "p388", "p389", "p390", "p391", "p392", "p393", "p394", "p395", "p396", "p397", "p398", "p399"],
    "openai": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    "edge": ["en-US-AriaNeural", "en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaRUS", "en-US-BenjaminRUS", "en-US-GuyRUS", "en-US-ZiraRUS", "en-US-JessaRUS", "en-US-Jessa24kRUS", "en-US-Sean", "en-US-Jason", "en-US-Cora", "en-US-Jane", "en-US-Tony", "en-US-Amber", "en-US-Ana", "en-US-Ashley", "en-US-Brandon", "en-US-Christopher", "en-US-Davis", "en-US-Elizabeth", "en-US-Jacob", "en-US-JennyMultilingualNeural", "en-US-Michelle", "en-US-Monica", "en-US-Roger", "en-US-Steffan", "en-US-AndrewNeural", "en-US-EmmaNeural", "en-US-BrianNeural", "en-US-AriaNeural", "en-US-JennyNeural", "en-US-GuyNeural"],
    "kokoro": ["af_sky", "af_bella", "af_sarah", "am_adam", "bf_emma", "bm_george", "af_nicole", "am_michael", "af_sky", "af_bella", "af_sarah", "am_adam", "bf_emma", "bm_george", "af_nicole", "am_michael"],
    "xtts": ["Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema", "Alison Dietlinde", "Ana Florence", "Annmarie Nele", "Asya Anara", "Brenda Stern", "Gitta Nikolina", "Henriette Usha", "Sofia Hellen", "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie", "Andrew Chipper", "Badr Odhiambo", "Dionisio Schuyler", "Royston Min", "Viktor Eka", "Abrahan Mack", "Adde Michal", "Baldur Sanjin", "Craig Gutsy", "Damien Black", "Gilberto Mathias", "Ilkin Urbano", "Kazuhiko Atallah", "Ludvig Milivoj", "Suad Qasim", "Torcull Diarmuid", "Viktor Menelaos", "Zacharie Aimilios"]
}

class Epub2TTSInterface:
    def __init__(self):
        self.conversion_thread = None
        self.output_queue = queue.Queue()
        
    def get_speakers(self, engine):
        """Return speakers for the selected engine"""
        return ENGINE_SPEAKERS.get(engine, [])
    
    def convert_epub(self, epub_file, engine, speaker, start_chapter, end_chapter, 
                    threads, output_format, bitrate, min_ratio, debug, skiplinks, 
                    skipfootnotes, sayparts, no_deepspeed, skip_cleanup, openai_key, 
                    xtts_samples, speed, progress=gr.Progress()):
        """Main conversion function"""
        
        if not epub_file:
            return "Error: Please select an EPUB file"
        
        # Clean up any existing temporary files
        self.cleanup_temp_files()
        
        try:
            # Handle file upload
            if hasattr(epub_file, 'name'):
                epub_path = epub_file.name
            else:
                epub_path = epub_file
            
            # Validate file
            if not os.path.exists(epub_path):
                return f"Error: File {epub_path} not found"
            
            if not epub_path.lower().endswith(('.epub', '.txt')):
                return "Error: Please select a valid EPUB or TXT file"
            
            # Prepare arguments
            args = argparse.Namespace()
            args.sourcefile = epub_path
            args.engine = engine
            args.speaker = speaker
            args.start = start_chapter
            args.threads = threads
            args.end = end_chapter
            args.language = "en"
            args.minratio = min_ratio
            args.model = "tts_models/en/vctk/vits"
            args.debug = debug
            args.skiplinks = skiplinks
            args.skipfootnotes = skipfootnotes
            args.sayparts = sayparts
            args.audioformat = output_format
            args.bitrate = bitrate
            args.no_deepspeed = no_deepspeed
            args.skip_cleanup = skip_cleanup
            args.openai = openai_key if engine == "openai" else None
            args.xtts = xtts_samples if engine == "xtts" else None
            args.speed = speed
            args.scan = False
            args.export = None
            args.cover = None
            
            # Capture output
            output_buffer = io.StringIO()
            
            # Create audiobook
            mybook = EpubToAudiobook(
                source=args.sourcefile,
                start=args.start,
                threads=args.threads,
                end=args.end,
                skiplinks=args.skiplinks,
                engine=args.engine,
                minratio=args.minratio,
                model_name=args.model,
                debug=args.debug,
                language=args.language,
                skipfootnotes=args.skipfootnotes,
                sayparts=args.sayparts,
                no_deepspeed=args.no_deepspeed,
                skip_cleanup=args.skip_cleanup,
                audioformat=args.audioformat,
                speed=args.speed,
            )
            
            # Get chapters
            if mybook.sourcetype == "epub":
                mybook.get_chapters_epub(speaker=speaker)
            else:
                mybook.get_chapters_text(speaker=speaker)
            
            # Update end chapter if needed
            if end_chapter == 999:
                end_chapter = len(mybook.chapters_to_read)
            
            # Check if we need to overwrite existing files
            book_name = os.path.splitext(os.path.basename(epub_path))[0]
            voice_suffix = f"-{speaker.replace(' ', '-').lower()}"
            output_filename = f"{book_name}{voice_suffix}.m4b"
            
            # Clean up any existing partial files
            self.cleanup_existing_files(book_name, speaker)
            
            # Start conversion
            with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                mybook.read_book(
                    voice_samples=args.xtts,
                    engine=args.engine,
                    openai=args.openai,
                    model_name=args.model,
                    speaker=speaker,
                    bitrate=args.bitrate,
                )
            
            output = output_buffer.getvalue()
            
            # Check if output file was created
            if os.path.exists(output_filename):
                return f"Conversion completed successfully!\n\nOutput file: {output_filename}\n\nLog:\n{output}"
            else:
                return f"Conversion may have failed. Check log:\n{output}"
                
        except Exception as e:
            return f"Error during conversion: {str(e)}\n\nPlease check the log above for details."
    
    def cleanup_temp_files(self):
        """Clean up temporary files from previous runs"""
        temp_files = [f for f in os.listdir('.') if f.startswith('temp') and f.endswith('.wav')]
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
    
    def cleanup_existing_files(self, book_name, speaker):
        """Clean up existing chapter files for this book"""
        voice_suffix = f"-{speaker.replace(' ', '-').lower()}"
        output_filename = f"{book_name}{voice_suffix}.m4b"
        
        # Clean up chapter files
        chapter_files = [f for f in os.listdir('.') if f.startswith(book_name) and f.endswith('.wav')]
        for chapter_file in chapter_files:
            try:
                os.remove(chapter_file)
                timing_file = f"{chapter_file}.timing"
                if os.path.exists(timing_file):
                    os.remove(timing_file)
            except:
                pass
        
        # Clean up final output
        if os.path.exists(output_filename):
            try:
                os.remove(output_filename)
            except:
                pass

def create_interface():
    """Create and configure the Gradio interface"""
    
    interface = Epub2TTSInterface()
    
    with gr.Blocks(title="EPUB to Audiobook Converter", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📚 EPUB to Audiobook Converter")
        gr.Markdown("Convert EPUB files to audiobooks using various TTS engines")
        
        with gr.Row():
            with gr.Column(scale=2):
                # File upload
                epub_file = gr.File(
                    label="Upload EPUB or TXT file",
                    file_types=[".epub", ".txt"],
                    file_count="single"
                )
                
                # Engine selection
                engine = gr.Dropdown(
                    choices=list(ENGINE_SPEAKERS.keys()),
                    value="tts",
                    label="TTS Engine",
                    info="Select the text-to-speech engine"
                )
                
                # Speaker selection (dynamic)
                speaker = gr.Dropdown(
                    choices=ENGINE_SPEAKERS["tts"],
                    value="p335",
                    label="Speaker Voice",
                    info="Select the voice/speaker"
                )
                
                # Engine-specific options
                with gr.Row():
                    openai_key = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        visible=False,
                        placeholder="sk-..."
                    )
                    
                    xtts_samples = gr.Textbox(
                        label="XTTS Voice Samples",
                        visible=False,
                        placeholder="path/to/sample1.wav,path/to/sample2.wav"
                    )
                
                # Basic settings
                with gr.Row():
                    start_chapter = gr.Number(
                        label="Start Chapter",
                        value=1,
                        minimum=1,
                        step=1
                    )
                    end_chapter = gr.Number(
                        label="End Chapter",
                        value=999,
                        minimum=1,
                        step=1,
                        info="999 = all chapters"
                    )
                
                with gr.Row():
                    threads = gr.Number(
                        label="Threads",
                        value=1,
                        minimum=1,
                        maximum=8,
                        step=1
                    )
                    speed = gr.Number(
                        label="Reading Speed",
                        value=1.3,
                        minimum=0.5,
                        maximum=2.0,
                        step=0.1,
                        visible=False
                    )
                
                # Output settings
                output_format = gr.Radio(
                    choices=["m4b", "wav", "flac"],
                    value="m4b",
                    label="Output Format"
                )
                
                bitrate = gr.Textbox(
                    label="Bitrate",
                    value="69k",
                    placeholder="69k, 128k, 192k"
                )
                
                # Advanced options
                with gr.Accordion("Advanced Options", open=False):
                    min_ratio = gr.Number(
                        label="Minimum Match Ratio",
                        value=88,
                        minimum=0,
                        maximum=100,
                        step=1,
                        info="0 to disable whisper verification"
                    )
                    
                    with gr.Row():
                        debug = gr.Checkbox(label="Debug Mode", value=False)
                        skiplinks = gr.Checkbox(label="Skip Links", value=False)
                        skipfootnotes = gr.Checkbox(label="Skip Footnotes", value=False)
                    
                    with gr.Row():
                        sayparts = gr.Checkbox(label="Say Part Numbers", value=False)
                        no_deepspeed = gr.Checkbox(label="Disable DeepSpeed", value=False)
                        skip_cleanup = gr.Checkbox(label="Skip Text Cleanup", value=False)
                
                # Convert button
                convert_btn = gr.Button("Convert to Audiobook", variant="primary")
                
            with gr.Column(scale=3):
                # Output
                output = gr.Textbox(
                    label="Conversion Log",
                    lines=20,
                    max_lines=30,
                    interactive=False
                )
                
                # Download button (will be updated after conversion)
                download_btn = gr.File(label="Download Audiobook", visible=False)
        
        # Dynamic updates
        def update_speakers(engine):
            speakers = interface.get_speakers(engine)
            return gr.Dropdown(choices=speakers, value=speakers[0] if speakers else None)
        
        def update_engine_options(engine):
            show_openai = engine == "openai"
            show_xtts = engine == "xtts"
            show_speed = engine == "kokoro"
            return [
                gr.Textbox(visible=show_openai),
                gr.Textbox(visible=show_xtts),
                gr.Number(visible=show_speed)
            ]
        
        # Event handlers
        engine.change(
            fn=update_speakers,
            inputs=[engine],
            outputs=[speaker]
        )
        
        engine.change(
            fn=update_engine_options,
            inputs=[engine],
            outputs=[openai_key, xtts_samples, speed]
        )
        
        convert_btn.click(
            fn=interface.convert_epub,
            inputs=[
                epub_file, engine, speaker, start_chapter, end_chapter,
                threads, output_format, bitrate, min_ratio, debug, skiplinks,
                skipfootnotes, sayparts, no_deepspeed, skip_cleanup, openai_key,
                xtts_samples, speed
            ],
            outputs=[output]
        )
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
