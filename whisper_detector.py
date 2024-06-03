import logging
from io import BytesIO
from typing import Dict, List
import whisper
import tempfile
import os

# Constants
MODEL_NAME = "small"  # Hardcoded whisper model name

class WhisperDetector:
    def __init__(self):
        self.model = whisper.load_model(MODEL_NAME)

    def save_to_temp_file(self, audio_file: BytesIO) -> str:
        """
        - Saves BytesIO object to a temporary file
        Args:
        - audio_file: BytesIO object of the audio file
        Returns:
        - Path to the temporary file
        """
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(audio_file.read())
            return temp_file.name

    def detect_languages(self, audio_file: BytesIO) -> List[Dict[str, float]]:
        """
        - Detects the top 3 languages spoken in the audio file
        Args:
        - audio_file: BytesIO object of the audio file
        Returns:
        - List of dictionaries containing the top 3 languages and their probabilities
        """
        # Save BytesIO to a temporary file
        temp_file_path = self.save_to_temp_file(audio_file)
        
        try:
            # Load audio using Whisper's load_audio function
            audio = whisper.load_audio(temp_file_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            _, probs_dict = self.model.detect_language(mel)
            
            # Add logging to debug the structure of probs_dict
            logging.debug(f"probs_dict: {probs_dict}")
            
            # Ensure probs_dict is a dictionary
            if not isinstance(probs_dict, dict):
                raise ValueError("probs_dict is not a dictionary")
            
            top_3_languages = sorted(probs_dict.items(), key=lambda item: item[1], reverse=True)[:3]
            return [{"language": lang, "probability": prob} for lang, prob in top_3_languages]
        finally:
            # Clean up the temporary file
            os.remove(temp_file_path)