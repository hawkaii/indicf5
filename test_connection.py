import socket
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server_ip = "34.177.101.103"
server_port = 9998

def test_connection():
    """Test basic socket connection to the server."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10)
    
    try:
        logger.info(f"=" * 60)
        logger.info(f"Testing connection to {server_ip}:{server_port}")
        logger.info(f"=" * 60)
        
        # Test 1: Connection
        logger.info("Test 1: Attempting to connect...")
        start_time = time.time()
        s.connect((server_ip, server_port))
        connect_time = time.time() - start_time
        logger.info(f"✓ Connected successfully! (took {connect_time:.2f}s)")
        
        # Test 2: Send data
        test_msg = "कस्टमर को तभी कॉल करो जब ड्यूटी लेनी हो"
        logger.info(f"Test 2: Sending test message ({len(test_msg)} chars)...")
        data_to_send = test_msg.encode('utf-8')
        logger.info(f"  Encoded to {len(data_to_send)} bytes")
        
        s.sendall(data_to_send)
        logger.info("✓ Data sent successfully!")
        
        # Test 3: Signal end of sending
        logger.info("Test 3: Shutting down write channel...")
        s.shutdown(socket.SHUT_WR)
        logger.info("✓ Write channel closed")
        
        # Test 4: Wait for response
        logger.info("Test 4: Waiting for server response (max 30s)...")
        s.settimeout(30)
        
        response_chunks = 0
        total_bytes = 0
        start_receive = time.time()
        
        while True:
            data = s.recv(8192)
            if not data:
                logger.info("✓ Connection closed by server (no more data)")
                break
            
            if data == b"END":
                logger.info("✓ Received END signal from server")
                break
            
            response_chunks += 1
            total_bytes += len(data)
            
            if response_chunks == 1:
                first_chunk_time = time.time() - start_receive
                logger.info(f"✓ First chunk received! (latency: {first_chunk_time:.2f}s, size: {len(data)} bytes)")
            
            if response_chunks % 10 == 0:
                logger.info(f"  Received {response_chunks} chunks, {total_bytes} bytes so far...")
        
        total_time = time.time() - start_receive
        logger.info(f"✓ Received {response_chunks} chunks, {total_bytes} total bytes in {total_time:.2f}s")
        
        logger.info("=" * 60)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("=" * 60)
        
    except socket.timeout:
        logger.error("✗ Connection timed out - server not responding")
        logger.error("  Possible issues:")
        logger.error("  - Server is not running")
        logger.error("  - Firewall blocking connection")
        logger.error("  - Server processing is too slow")
        
    except ConnectionRefusedError:
        logger.error(f"✗ Connection refused to {server_ip}:{server_port}")
        logger.error("  Possible issues:")
        logger.error("  - Server is not running")
        logger.error("  - Wrong IP or port")
        logger.error("  - Firewall blocking connection")
        
    except Exception as e:
        logger.error(f"✗ Error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        s.close()
        logger.info("Connection closed")


if __name__ == "__main__":
    test_connection()
