import io
import base64
import torch
import tempfile
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel

# Initialize FastAPI app
app = FastAPI(title="IndicF5 TTS API", description="Text-to-Speech API using IndicF5 model")

# Reference audio path and text (hardcoded)
REF_AUDIO_PATH = "part3.wav"
REF_TEXT = "कस्टमर को तभी कॉल करो जब ड्यूटी लेनी हो; इससे रेटिंग अच्छी रहेगी और आगे ज़्यादा ड्यूटी मिलेगी—तैयार हो तो 'कॉल कस्टमर' दबाओ।"

# Load TTS model
print("Loading IndicF5 model...")
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = model.to(device)
print("Model loaded successfully!")

# Request model
class TTSRequest(BaseModel):
    text: str

# Response model
class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int

def synthesize_speech(text: str) -> tuple:
    """
    Synthesize speech from text using the reference audio and text.
    Returns tuple of (sample_rate, audio_data)
    """
    if not text or text.strip() == "":
        raise ValueError("Text cannot be empty")
    
    try:
        # Generate audio using the model
        audio = model(text, ref_audio_path=REF_AUDIO_PATH, ref_text=REF_TEXT)
        
        # Normalize output if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        return 24000, audio
    except Exception as e:
        raise Exception(f"Error during speech synthesis: {str(e)}")

@app.post("/synthesize", response_model=TTSResponse)
async def synthesize(request: TTSRequest):
    """
    Generate speech from input text.
    
    - **text**: The text to convert to speech
    
    Returns base64 encoded audio and sample rate.
    """
    try:
        # Synthesize speech
        sample_rate, audio_data = synthesize_speech(request.text)
        
        # Convert audio to WAV format in memory
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, samplerate=sample_rate, format='WAV')
        audio_buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return TTSResponse(
            audio_base64=audio_base64,
            sample_rate=sample_rate
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "IndicF5 TTS API",
        "version": "1.0",
        "endpoints": {
            "/synthesize": "POST - Generate speech from text",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "model": "ai4bharat/IndicF5",
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
