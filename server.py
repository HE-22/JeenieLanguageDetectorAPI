import os
from flask import Flask, request, jsonify
import logging
from io import BytesIO
from whisper_detector import WhisperDetector

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
detector = WhisperDetector()

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
            detected_languages = detector.detect_languages(audio_file)
            logging.info(f"Detected languages: {detected_languages}")

            return jsonify({"detected_languages": detected_languages}), 200
        except AttributeError as e:
            logging.error(f"AttributeError: {e}")
            return jsonify({"error": "Invalid file format or processing error"}), 400
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            return jsonify({"error": "Error processing file"}), 500

    logging.error("Unexpected error")
    return jsonify({"error": "Unexpected error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    app.run(host='0.0.0.0', port=port, debug=True)