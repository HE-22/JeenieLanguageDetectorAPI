from typing import Dict, List
import whisper
import logging
from flask import Flask, request, jsonify
from io import BytesIO
import tempfile
import os

# Constants
MODEL_NAME = "small"  # Hardcoded whisper model name

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

def detect_languages(audio_file: BytesIO) -> List[Dict[str, float]]:
    """
    - Detects the top 3 languages spoken in the audio file
    Args:
    - audio_file: BytesIO object of the audio file
    Returns:
    - List of dictionaries containing the top 3 languages and their probabilities
    """
    # Load the Whisper model
    model = whisper.load_model(MODEL_NAME)
    
    # Save BytesIO to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(audio_file.read())
        temp_file_path = temp_file.name
    
    try:
        # Load audio using Whisper's load_audio function
        audio = whisper.load_audio(temp_file_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        top_3_languages = sorted(probs.items(), key=lambda item: item[1], reverse=True)[:3]
        return [{"language": lang, "probability": prob} for lang, prob in top_3_languages]
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

@app.route('/detect_languages', methods=['POST'])
def detect_languages_api():
    """
    - API endpoint to detect top languages from an uploaded audio file
    """
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # Read the file into a BytesIO object
            audio_file = BytesIO(file.read())
            
            # Detect languages
            detected_languages = detect_languages(audio_file)
            logging.info(f"Detected languages: {detected_languages}")

            return jsonify({"detected_languages": detected_languages}), 200
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            return jsonify({"error": "Error processing file"}), 500

    logging.error("Unexpected error")
    return jsonify({"error": "Unexpected error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)


