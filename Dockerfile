# Use official IndicF5 image as base (includes all dependencies and the model)
FROM registry.hf.space/ai4bharat-indicf5:latest

# Set working directory
WORKDIR /app

# Install additional dependencies for FastAPI
RUN pip3 install --no-cache-dir fastapi uvicorn[standard] pydantic

# Copy application files
COPY api.py .
COPY part3.wav .

# Expose port 8000 for the API
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the API application
CMD ["python3", "api.py"]
