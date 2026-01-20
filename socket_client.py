import asyncio
import logging
import socket
import time

import numpy as np
import pyaudio


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def listen_to_IndicF5(text, server_ip="34.177.101.103", server_port=9998):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(30)  # 30 second timeout
    
    logger.info(f"Connecting to {server_ip}:{server_port}...")
    await asyncio.get_event_loop().run_in_executor(
        None, client_socket.connect, (server_ip, int(server_port))
    )
    logger.info("✓ Connected successfully!")
    
    # Enable TCP_NODELAY for immediate sending
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    start_time = time.time()
    first_chunk_time = None

    async def play_audio_stream():
        nonlocal first_chunk_time
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=24000,
            output=True,
            frames_per_buffer=2048,
        )

        try:
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
                
                # Only process if data length is a multiple of 4 (float32 size)
                if len(data) % 4 != 0:
                    # Trim to the nearest multiple of 4
                    remainder = len(data) % 4
                    logger.warning(f"⚠ Trimming {remainder} bytes from chunk to align to float32")
                    data = data[:-remainder]
                
                if len(data) > 0:
                    audio_array = np.frombuffer(data, dtype=np.float32)
                    stream.write(audio_array.tobytes())

                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        latency = first_chunk_time - start_time
                        logger.info(f"✓ First chunk received! (latency: {latency:.2f}s)")
                
                # If we found END signal, stop receiving
                if end_signal_received:
                    break

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        logger.info(f"Total time taken: {time.time() - start_time:.4f} seconds")

    try:
        data_to_send = f"{text}".encode("utf-8")
        logger.info(f"Sending text ({len(data_to_send)} bytes): {text[:50]}...")
        await asyncio.get_event_loop().run_in_executor(
            None, client_socket.sendall, data_to_send
        )
        logger.info("✓ Text sent successfully, waiting for audio response...")
        await play_audio_stream()

    except Exception as e:
        logger.error(f"Error in listen_to_IndicF5: {e}")
        import traceback
        traceback.print_exc()

    finally:
        client_socket.close()
        logger.info("Connection closed")


if __name__ == "__main__":
    text_to_send = "कस्टमर को तभी कॉल करो जब ड्यूटी लेनी हो; इससे रेटिंग अच्छी रहेगी और आगे ज़्यादा ड्यूटी मिलेगी।"

    asyncio.run(listen_to_IndicF5(text_to_send))
