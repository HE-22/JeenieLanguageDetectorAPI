FROM python:3.12.1-slim

# Copy local code to the container image
COPY . /app

# Set the working directory
WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable
ENV PORT 8080
ENV FLASK_APP=main.py

# Run the web service on container startup
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
