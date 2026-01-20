import asyncio
import logging
import socket
import time
import wave
from datetime import datetime

import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def receive_and_save_audio(text, server_ip="34.177.101.103", server_port=9998, output_file=None):
    """
    Connect to IndicF5 socket server, send text, and save received audio to WAV file.
    
    Args:
        text: Text to synthesize
        server_ip: Server IP address
        server_port: Server port
        output_file: Output WAV file path (auto-generated if None)
    """
    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output_{timestamp}.wav"
    
    # Socket setup
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(30)  # 30 second timeout
    
    logger.info(f"Connecting to {server_ip}:{server_port}...")
    
    try:
        # Connect to server
        await asyncio.get_event_loop().run_in_executor(
            None, client_socket.connect, (server_ip, int(server_port))
        )
        logger.info(f"✓ Connected successfully!")
        
        # Enable TCP_NODELAY for immediate sending
        client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        start_time = time.time()
        first_chunk_time = None
        
        # Send text to server
        data_to_send = text.encode("utf-8")
        logger.info(f"Sending text ({len(data_to_send)} bytes): {text[:50]}...")
        await asyncio.get_event_loop().run_in_executor(
            None, client_socket.sendall, data_to_send
        )
        logger.info("✓ Text sent successfully, waiting for audio response...")
        
        # Receive and save audio
        audio_chunks = []
        chunk_count = 0
        total_bytes = 0
        
        # WAV file parameters
        sample_rate = 24000
        channels = 1
        sample_width = 4  # float32 = 4 bytes
        
        logger.info(f"Receiving audio and saving to {output_file}...")
        
        end_signal_received = False
        
        while True:
            data = await asyncio.get_event_loop().run_in_executor(
                None, client_socket.recv, 8192
            )
            
            if not data:
                logger.info("✓ Connection closed by server")
                break
            
            if data == b"END":
                logger.info("✓ Received END signal from server")
                break
            
            # Check if data contains END signal mixed with audio data
            if b"END" in data:
                logger.info("✓ Received END signal (mixed with data)")
                end_signal_received = True
                # Extract only the audio data before END
                end_pos = data.find(b"END")
                data = data[:end_pos]
            
            # Skip empty data
            if len(data) == 0:
                break
            
            # Convert received bytes to float32 audio
            # Only process if data length is a multiple of 4 (float32 size)
            if len(data) % 4 != 0:
                # Trim to the nearest multiple of 4
                remainder = len(data) % 4
                logger.warning(f"⚠ Trimming {remainder} bytes from chunk to align to float32")
                data = data[:-remainder]
            
            if len(data) > 0:
                audio_array = np.frombuffer(data, dtype=np.float32)
                audio_chunks.append(audio_array)
                
                chunk_count += 1
                total_bytes += len(data)
                
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    latency = first_chunk_time - start_time
                    logger.info(f"✓ First chunk received! (latency: {latency:.2f}s, size: {len(data)} bytes)")
                
                if chunk_count % 10 == 0:
                    logger.info(f"  Received {chunk_count} chunks, {total_bytes} bytes...")
            
            # If we found END signal, stop receiving
            if end_signal_received:
                break
        
        # Combine all audio chunks
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            logger.info(f"✓ Total audio samples: {len(full_audio)} ({len(full_audio)/sample_rate:.2f} seconds)")
            
            # Convert float32 (-1.0 to 1.0) to int16 for WAV file
            audio_int16 = np.int16(full_audio * 32767)
            
            # Write to WAV file
            with wave.open(output_file, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            total_time = time.time() - start_time
            file_size = len(audio_int16.tobytes()) / 1024  # KB
            
            logger.info("=" * 60)
            logger.info(f"✓ SUCCESS! Audio saved to: {output_file}")
            logger.info(f"  Duration: {len(full_audio)/sample_rate:.2f} seconds")
            logger.info(f"  File size: {file_size:.1f} KB")
            logger.info(f"  Total time: {total_time:.2f} seconds")
            logger.info(f"  Chunks received: {chunk_count}")
            logger.info("=" * 60)
        else:
            logger.warning("⚠ No audio data received from server")
    
    except socket.timeout:
        logger.error("✗ Connection timed out - server not responding")
    except ConnectionRefusedError:
        logger.error(f"✗ Connection refused to {server_ip}:{server_port}")
    except Exception as e:
        logger.error(f"✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client_socket.close()
        logger.info("Connection closed")


if __name__ == "__main__":
    # Example text to synthesize
    text_to_send = "कस्टमर को तभी कॉल करो जब ड्यूटी लेनी हो; इससे रेटिंग अच्छी रहेगी और आगे ज़्यादा ड्यूटी मिलेगी।"
    
    # You can optionally specify a custom output filename
    # output_filename = "my_custom_audio.wav"
    # asyncio.run(receive_and_save_audio(text_to_send, output_file=output_filename))
    
    asyncio.run(receive_and_save_audio(text_to_send))
