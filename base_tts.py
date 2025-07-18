import re
import whisper
from fuzzywuzzy import fuzz

class Text2WaveFile:
    whispermodel = None
    debug = False
    def __init__(self, config = {}):
        """
        initalizes a Text 2 Wave File class
        This might mean loading the ML model used for speech syntesis or setting up other stuff
        """
        self.config = config

    def proccess_text(self, text, wave_file_name):
        """
        takes a pice of text and generates audio from it then saves that audio in wave_file_name
        returns True if successfull
        """
        pass

    def compare(self, text, wavfile):
        if self.whispermodel is None:
            self.whispermodel = whisper.load_model("tiny")
        
        result = self.whispermodel.transcribe(wavfile)
        text = re.sub(" +", " ", text).lower().strip()
        ratio = fuzz.ratio(text, result["text"].lower())
        print(f"Transcript: {result['text'].lower()}") if self.debug else None
        print(f"Text to transcript comparison ratio: {ratio}") if self.debug else None
        return ratio, result['text']

    
    def proccess_text_retry(self, text, wave_file_name):
        retries = 2
        while retries > 0:
            self.proccess_text(text, wave_file_name)
            result_text = ""
            if self.config['minratio'] == 0:
                print("Skipping whisper transcript comparison") if self.config['debug'] else None
                ratio = self.config['minratio']
            else:
                ratio, result_text = self.compare(text, wave_file_name)
            if ratio < self.config['minratio']:
                print(f"Spoken text did not sound right after control with whisper - {ratio}\nInput: {text}\nOutput: {result_text}")
            else:
                break
            retries -= 1
        if retries == 0:
            print(f"Something is wrong with the audio acording to whisper ({ratio}): {wave_file_name}")
