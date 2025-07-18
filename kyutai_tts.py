#!/usr/bin/env python3
"""
Kyutai TTS integration for epub2tts
Based on the provided kyutai_tts.py example
"""

import numpy as np
import torch
import soundfile as sf
import os
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO, TTSModel
from epub2tts import Text2WaveFile

class KyutaiTTS(Text2WaveFile):
    def __init__(self, config={}):
        """
        Initialize Kyutai TTS
        
        Args:
            config: Configuration dictionary with keys:
                - voice: path to voice file (default: "expresso/ex03-ex01_happy_001_channel1_334s.wav")
                - device: "cuda" or "cpu" (default: auto-detect)
                - temp: temperature for generation (default: 0.6)
                - cfg_coef: CFG coefficient (default: 2.0)
                - n_q: number of quantizers (default: 32)
        """
        if 'voice' not in config:
            config['voice'] = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
        print(f"Voice:{config['voice']}")
        self.config = config
        
        # Auto-detect device
        if 'device' not in config:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config['device'])
        
        self.temp = config.get('temp', 0.6)
        self.cfg_coef = config.get('cfg_coef', 2.0)
        self.n_q = config.get('n_q', 32)
        
        print(f"Loading Kyutai TTS on {self.device}...")
        
        try:
            checkpoint_info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
            self.tts_model = TTSModel.from_checkpoint_info(
                checkpoint_info, 
                n_q=self.n_q, 
                temp=self.temp, 
                device=self.device
            )
            self.voice_path = self.tts_model.get_voice_path(config['voice'])
            print(f"Kyutai TTS loaded successfully with voice: {config['voice']}")
        except Exception as e:
            raise Exception(f"Failed to load Kyutai TTS: {str(e)}")

    def proccess_text(self, text, wave_file_name):
        """
        Convert text to speech using Kyutai TTS
        
        Args:
            text: Text to synthesize
            wave_file_name: Output file path
            
        Returns:
            bool: True if successful
        """
        try:
            # Prepare the script
            entries = self.tts_model.prepare_script([text], padding_between=1)
            
            # Create condition attributes
            condition_attributes = self.tts_model.make_condition_attributes(
                [self.voice_path], cfg_coef=self.cfg_coef
            )
            
            # Generate audio
            pcms = []
            
            def _on_frame(frame):
                if (frame != -1).all():
                    pcm = self.tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                    pcms.append(np.clip(pcm[0, 0], -1, 1))
            
            # Generate
            all_entries = [entries]
            all_condition_attributes = [condition_attributes]
            
            with self.tts_model.mimi.streaming(len(all_entries)):
                result = self.tts_model.generate(
                    all_entries, 
                    all_condition_attributes, 
                    on_frame=_on_frame
                )
            
            # Concatenate and save
            audio = np.concatenate(pcms, axis=-1)
            sf.write(wave_file_name, audio, self.tts_model.mimi.sample_rate)
            
            return os.path.exists(wave_file_name)
            
        except Exception as e:
            print(f"Error in Kyutai TTS: {str(e)}")
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
