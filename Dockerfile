# Use an official Python runtime as a parent image
FROM python:3.12.1-slim

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download and cache the Whisper model
RUN python -c "import whisper; model = whisper.load_model('large-v2'); model.save_pretrained('/app/whisper_model')"

# Define environment variable
ENV FLASK_APP=main.py

# Run Gunicorn server when the container launches
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
