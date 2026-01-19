import argparse
import gc
import logging
import queue
import socket
import struct
import threading
import traceback
import wave
import tempfile

import numpy as np
import torch
import torchaudio
import soundfile as sf
from transformers import AutoModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set torchaudio backend to soundfile instead of torchcodec
torchaudio.set_audio_backend("soundfile")
logger.info("Set torchaudio backend to soundfile")

# Reference audio path and text (hardcoded like api.py)
REF_AUDIO_PATH = "part3.wav"
REF_TEXT = "कस्टमर को तभी कॉल करो जब ड्यूटी लेनी हो; इससे रेटिंग अच्छी रहेगी और आगे ज़्यादा ड्यूटी मिलेगी—तैयार हो तो 'कॉल कस्टमर' दबाओ।"

# Monkey patch to fix torch.no_available_grad bug in IndicF5
if not hasattr(torch, 'no_available_grad'):
    torch.no_available_grad = torch.no_grad
    logger.info("Applied monkey patch for torch.no_available_grad -> torch.no_grad")


class AudioFileWriterThread(threading.Thread):
    """Threaded file writer to avoid blocking the TTS streaming process."""

    def __init__(self, output_file, sampling_rate):
        super().__init__()
        self.output_file = output_file
        self.sampling_rate = sampling_rate
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.audio_data = []

    def run(self):
        """Process queued audio data and write it to a file."""
        logger.info("AudioFileWriterThread started.")
        with wave.open(self.output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sampling_rate)

            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    chunk = self.queue.get(timeout=0.1)
                    if chunk is not None:
                        chunk = np.int16(chunk * 32767)
                        self.audio_data.append(chunk)
                        wf.writeframes(chunk.tobytes())
                except queue.Empty:
                    continue

    def add_chunk(self, chunk):
        """Add a new chunk to the queue."""
        self.queue.put(chunk)

    def stop(self):
        """Stop writing and ensure all queued data is written."""
        self.stop_event.set()
        self.join()
        logger.info("Audio writing completed.")


class TTSStreamingProcessor:
    def __init__(self, device=None, dtype=torch.float32):
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.dtype = dtype
        
        # Load IndicF5 model
        logger.info("Loading IndicF5 model...")
        repo_id = "ai4bharat/IndicF5"
        self.model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
        self.model = self.model.to(self.device)
        logger.info(f"Model loaded successfully on device: {self.device}")
        
        # Reference audio settings
        self.ref_audio_path = REF_AUDIO_PATH
        self.ref_text = REF_TEXT
        self.sampling_rate = 24000  # IndicF5 output sample rate
        
        # Verify reference audio exists
        try:
            ref_audio_data, ref_sample_rate = sf.read(self.ref_audio_path)
            logger.info(f"Reference audio loaded: {self.ref_audio_path} ({ref_sample_rate} Hz)")
        except Exception as e:
            logger.error(f"Failed to load reference audio: {e}")
            raise
        
        self._warm_up()
        self.file_writer_thread = None

    def _warm_up(self):
        """Warm up the model with a test synthesis."""
        logger.info("Warming up the model...")
        try:
            # Load reference audio for warm-up
            ref_audio_data, ref_sample_rate = sf.read(self.ref_audio_path)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                sf.write(temp_audio.name, ref_audio_data, samplerate=ref_sample_rate, format='WAV')
                temp_audio.flush()
                temp_ref_path = temp_audio.name
            
            # Warm-up synthesis
            gen_text = "यह एक टेस्ट है।"
            _ = self.model(gen_text, ref_audio_path=temp_ref_path, ref_text=self.ref_text)
            logger.info("Warm-up completed.")
        except Exception as e:
            logger.warning(f"Warm-up failed: {e}")

    def generate_stream(self, text, conn):
        """Generate audio and stream it in chunks to the client."""
        if not text or text.strip() == "":
            logger.warning("Empty text received, skipping generation")
            conn.sendall(b"END")
            return
        
        try:
            # Load reference audio
            logger.info(f"Loading reference audio: {self.ref_audio_path}")
            ref_audio_data, ref_sample_rate = sf.read(self.ref_audio_path)
            
            # Save reference audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                sf.write(temp_audio.name, ref_audio_data, samplerate=ref_sample_rate, format='WAV')
                temp_audio.flush()
                temp_ref_path = temp_audio.name
            
            # Generate complete audio using IndicF5 model
            logger.info(f"Synthesizing text: {text[:50]}...")
            audio = self.model(text, ref_audio_path=temp_ref_path, ref_text=self.ref_text)
            
            # Normalize output if needed
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            
            logger.info(f"Generated complete audio of size: {len(audio)}")
            
            # Reset the file writer thread
            if self.file_writer_thread is not None:
                self.file_writer_thread.stop()
            self.file_writer_thread = AudioFileWriterThread(
                "output.wav", self.sampling_rate
            )
            self.file_writer_thread.start()
            
            # Stream audio in chunks (simulated streaming)
            chunk_size = 2048  # samples per chunk
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                
                if len(chunk) > 0:
                    logger.info(f"Streaming chunk {i//chunk_size + 1}: {len(chunk)} samples")
                    
                    # Send audio chunk via socket
                    conn.sendall(struct.pack(f"{len(chunk)}f", *chunk))
                    
                    # Write to file asynchronously
                    self.file_writer_thread.add_chunk(chunk)
            
            logger.info("Finished sending audio stream.")
            conn.sendall(b"END")  # Send end signal
            
            # Ensure all audio data is written before exiting
            self.file_writer_thread.stop()
            
        except Exception as e:
            logger.error(f"Error during synthesis: {e}")
            traceback.print_exc()
            conn.sendall(b"END")  # Send end signal even on error


def handle_client(conn, processor):
    try:
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                data_str = data.decode("utf-8").strip()
                logger.info(f"Received text: {data_str}")

                try:
                    processor.generate_stream(data_str, conn)
                except Exception as inner_e:
                    logger.error(f"Error during processing: {inner_e}")
                    traceback.print_exc()
                    break
    except Exception as e:
        logger.error(f"Error handling client: {e}")
        traceback.print_exc()


def start_server(host, port, processor):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        logger.info(f"Server started on {host}:{port}")
        while True:
            conn, addr = s.accept()
            logger.info(f"Connected by {addr}")
            handle_client(conn, processor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndicF5 TTS Streaming Socket Server")

    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", default=9998, type=int, help="Port to bind to")
    parser.add_argument("--device", default=None, help="Device to run the model on (cuda/cpu)")
    parser.add_argument(
        "--dtype", default="float32", help="Data type to use for model inference (float32/float16)"
    )

    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.float32)

    try:
        # Initialize the processor with the IndicF5 model
        processor = TTSStreamingProcessor(
            device=args.device,
            dtype=dtype,
        )

        # Start the server
        start_server(args.host, args.port, processor)

    except KeyboardInterrupt:
        gc.collect()
