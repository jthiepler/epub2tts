#!/usr/bin/env python3
"""
Dynamic Kyutai voice loader for epub2tts
Loads voices from kyutai-tts-voices directory with full directory traversal
"""

import os
from pathlib import Path

def get_kyutai_voices():
    """Get all available Kyutai voices by finding actual .wav files"""
    
    base_dir = Path(__file__).parent / "kyutai-tts-voices"
    voices = []
    
    if not base_dir.exists():
        return ["af", "am", "bf", "bm"]  # Fallback voices
    
    # Traverse all directories recursively to find .wav files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.wav'):
                # Get relative path from base directory
                full_path = Path(root) / file
                relative_path = full_path.relative_to(base_dir)
                voices.append(str(relative_path))
    
    # Also look for .safetensors files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.safetensors'):
                full_path = Path(root) / file
                relative_path = full_path.relative_to(base_dir)
                voices.append(str(relative_path))
    
    # Remove duplicates and sort
    voices = sorted(list(set(voices)))
    
    # If no voices found, return fallback
    if not voices:
        return ["af", "am", "bf", "bm"]
    
    return voices

def get_kyutai_voice_path(voice_name):
    """Return the voice path as-is since it's already in the correct format"""
    return voice_name

if __name__ == "__main__":
    voices = get_kyutai_voices()
    print(f"Found {len(voices)} Kyutai voices:")
    for voice in voices[:20]:  # Show first 20
        print(f"  - {voice}")
    if len(voices) > 20:
        print(f"  ... and {len(voices) - 20} more")
