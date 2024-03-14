import base64
import fcntl
import logging
import os
import socket
import tempfile
from pathlib import Path

import requests

set_motion_url = 'http://127.0.0.1:8000/set_motion'
background_dir = Path("background")
background_file_name = "bg-server.jpg"
background_file_path = background_dir / background_file_name

logging.basicConfig(level=logging.INFO)


def atomic_write(file_path, data):
    """
    Writes data to a file in an atomic manner.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=file_path.parent, mode="wb")
    temp_file.write(data)
    temp_file.close()
    os.rename(temp_file.name, file_path)


def udp_motion_receiver():
    localIP = "127.0.0.1"
    motionPort = 20003
    bgPort = 20004
    bufferSize = 4096

    bgSegLength = 200
    bgCache = bytearray()

    # Ensure background directory exists
    background_dir.mkdir(exist_ok=True)

    # UDP sockets setup
    UDPMotionSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPMotionSocket.bind((localIP, motionPort))
    fcntl.fcntl(UDPMotionSocket, fcntl.F_SETFL, os.O_NONBLOCK)
    logging.info("motion_receiver up and listening")

    UDPBgSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPBgSocket.bind((localIP, bgPort))
    logging.info("bg_receiver up and listening")

    while True:
        try:
            data, _ = UDPMotionSocket.recvfrom(bufferSize)
            response = requests.post(set_motion_url, json={"ldmk": base64.b64encode(data).decode('ascii')})
            logging.info(f"Server response: {response}")
        except BlockingIOError:
            pass  # No data received, non-blocking mode
        except Exception as e:
            logging.error(f"Error processing motion data: {e}")

        try:
            bgData, _ = UDPBgSocket.recvfrom(bufferSize)
            bgCache += bgData
            if len(bgData) < bgSegLength:
                atomic_write(background_file_path, bgCache)
                logging.info(f"Background updated, file length: {len(bgCache)}")
                bgCache.clear()
        except Exception as e:
            logging.error(f"Error receiving background data: {e}")


if __name__ == '__main__':
    udp_motion_receiver()
