# Use official IndicF5 image as base (includes all dependencies and the model)
FROM registry.hf.space/ai4bharat-indicf5:latest

# Set working directory
WORKDIR /app

# Install system dependencies for PyAudio
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    python3-pyaudio \
    && rm -rf /var/lib/apt/lists/*

# Install additional dependencies for FastAPI and socket server
RUN pip3 install --no-cache-dir fastapi uvicorn[standard] pydantic pyaudio

# Copy application files
COPY api.py .
COPY socket_server.py .
COPY socket_client.py .
COPY part3.wav .

# Expose port 8000 for the API
EXPOSE 8000

# Expose port 9998 for the socket server
EXPOSE 9998

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the API application (change to socket_server.py if needed)
CMD ["python3", "api.py"]
