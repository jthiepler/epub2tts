#!/usr/bin/env python3
"""
Kyutai MLX TTS integration for epub2tts
Based on the MLX implementation of Kyutai TTS
"""

import json
import os
import queue
import time
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece
import sphn
from moshi_mlx import models
from moshi_mlx.models.tts import (
    DEFAULT_DSM_TTS_REPO,
    DEFAULT_DSM_TTS_VOICE_REPO,
    TTSModel,
)
from moshi_mlx.utils.loaders import hf_get
from base_tts import Text2WaveFile


class KyutaiMLXTTS(Text2WaveFile):
    def __init__(self, config={}):
        """
        Initialize Kyutai MLX TTS
        
        Args:
            config: Configuration dictionary with keys:
                - voice: path to voice file (default: "expresso/ex03-ex01_happy_001_channel1_334s.wav")
                - quantize: quantization bits (None, 8, 4) (default: None)
                - temp: temperature for generation (default: 0.6)
                - cfg_coef: CFG coefficient (default: 1.0)
                - hf_repo: Hugging Face repo for models (default: DEFAULT_DSM_TTS_REPO)
                - voice_repo: Hugging Face repo for voices (default: DEFAULT_DSM_TTS_VOICE_REPO)
        """
        if 'voice' not in config:
            config['voice'] = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
        print(f"Voice: {config['voice']}")
        self.config = config
        
        self.temp = config.get('temp', 0.6)
        self.cfg_coef = config.get('cfg_coef', 1.0)
        quantize_str = config.get('quantize', 'None')
        if quantize_str == '8-bit':
            self.quantize = 8
        elif quantize_str == '4-bit':
            self.quantize = 4
        else:
            self.quantize = None
        self.hf_repo = config.get('hf_repo', DEFAULT_DSM_TTS_REPO)
        self.voice_repo = config.get('voice_repo', DEFAULT_DSM_TTS_VOICE_REPO)
        
        print("🍎 Initializing Kyutai MLX TTS...")
        
        try:
            self._load_model()
            print("Kyutai MLX TTS loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load Kyutai MLX TTS: {str(e)}")

    def _load_model(self):
        """Load the MLX model with optional quantization"""
        mx.random.seed(299792458)
        
        # Load configuration
        raw_config = hf_get("config.json", self.hf_repo)
        with open(hf_get(raw_config), "r") as fobj:
            raw_config = json.load(fobj)

        # Load model weights
        mimi_weights = hf_get(raw_config["mimi_name"], self.hf_repo)
        moshi_name = raw_config.get("moshi_name", "model.safetensors")
        moshi_weights = hf_get(moshi_name, self.hf_repo)
        tokenizer = hf_get(raw_config["tokenizer_name"], self.hf_repo)
        
        # Initialize model
        lm_config = models.LmConfig.from_config_dict(raw_config)
        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)

        # Load weights
        model.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)

        # Apply quantization if specified
        if self.quantize is not None:
            print(f"🔧 Quantizing model to {self.quantize} bits")
            nn.quantize(model.depformer, bits=self.quantize)
            for layer in model.transformer.layers:
                nn.quantize(layer.self_attn, bits=self.quantize)
                nn.quantize(layer.gating, bits=self.quantize)

        # Load tokenizers
        text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer))
        generated_codebooks = lm_config.generated_codebooks
        audio_tokenizer = models.mimi.Mimi(models.mimi_202407(generated_codebooks))
        audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)

        # Initialize TTS model
        cfg_coef_conditioning = None
        self.tts_model = TTSModel(
            model,
            audio_tokenizer,
            text_tokenizer,
            voice_repo=self.voice_repo,
            temp=self.temp,
            cfg_coef=1,
            max_padding=8,
            initial_padding=2,
            final_padding=2,
            padding_bonus=0,
            raw_config=raw_config,
        )
        
        if self.tts_model.valid_cfg_conditionings:
            cfg_coef_conditioning = self.tts_model.cfg_coef
            self.tts_model.cfg_coef = 1.0
            self.cfg_is_no_text = False
            self.cfg_is_no_prefix = False
        else:
            self.cfg_is_no_text = True
            self.cfg_is_no_prefix = True
            
        self.cfg_coef_conditioning = cfg_coef_conditioning

    def proccess_text(self, text, wave_file_name):
        """
        Convert text to speech using Kyutai MLX TTS
        
        Args:
            text: Text to synthesize
            wave_file_name: Output file path
            
        Returns:
            bool: True if successful
        """
        try:
            # Prepare the script
            all_entries = [self.tts_model.prepare_script([text])]
            
            # Handle voice selection
            if self.tts_model.multi_speaker:
                voices = [self.tts_model.get_voice_path(self.config['voice'])]
            else:
                voices = []
                
            # Create condition attributes
            all_attributes = [
                self.tts_model.make_condition_attributes(voices, self.cfg_coef_conditioning)
            ]

            # Audio frames queue
            wav_frames = queue.Queue()
            pcms = []

            def _on_frame(frame):
                """Callback for audio frames"""
                if (frame != -1).all():
                    pcm = self.tts_model.mimi.decode_step(frame[:, :, None])
                    pcm = np.array(mx.clip(pcm[0, 0], -1, 1))
                    wav_frames.put_nowait(pcm)
                    pcms.append(pcm)

            # Generate audio
            print("🎤 Generating audio...")
            begin = time.time()
            
            result = self.tts_model.generate(
                all_entries,
                all_attributes,
                cfg_is_no_prefix=self.cfg_is_no_prefix,
                cfg_is_no_text=self.cfg_is_no_text,
                on_frame=_on_frame,
            )
            
            # Calculate performance metrics
            frames = mx.concat(result.frames, axis=-1)
            total_duration = frames.shape[0] * frames.shape[-1] / self.tts_model.mimi.frame_rate
            time_taken = time.time() - begin
            total_speed = total_duration / time_taken
            
            print(f"✅ Generated audio in {time_taken:.2f}s ({total_speed:.2f}x real-time)")

            # Save audio
            if pcms:
                audio = np.concatenate(pcms, axis=-1)
                sphn.write_wav(wave_file_name, audio, self.tts_model.mimi.sample_rate)
                return os.path.exists(wave_file_name)
            else:
                print("❌ No audio generated")
                return False
                
        except Exception as e:
            print(f"❌ Error in Kyutai MLX TTS: {str(e)}")
            return False

    @staticmethod
    def list_available_voices():
        """List available voices from Hugging Face"""
        try:
            from huggingface_hub import list_repo_files
            files = list_repo_files(DEFAULT_DSM_TTS_VOICE_REPO)
            voice_files = [f for f in files if f.endswith('.wav')]
            return voice_files
        except Exception as e:
            print(f"Could not list voices: {str(e)}")
            return ["expresso/ex03-ex01_happy_001_channel1_334s.wav"]


# Example usage
if __name__ == "__main__":
    # Test the MLX implementation
    tts = KyutaiMLXTTS({
        'voice': 'expresso/ex03-ex01_happy_001_channel1_334s.wav',
        'quantize': 8  # Optional: 8-bit quantization
    })
    
    success = tts.proccess_text("Hello, this is a test of the Kyutai MLX TTS system.", "test_output.wav")
    print(f"Success: {success}")
