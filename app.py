import base64
import logging
import pickle
from collections import deque

import numpy as np
from flask import Flask, send_from_directory, jsonify, request

from mocap_util.motion_handler import motion_handler

# Configure logging
logging.basicConfig(encoding="utf-8", level=logging.DEBUG)

# Initialize Flask application
app = Flask(__name__)

# Define a deque to hold the last 10 landmarks
landmarks_queue = deque(maxlen=10)
received_landmarks = None


# Route definitions
@app.route('/')
def index():
    """Serve the main HTML file."""
    return send_from_directory('web', 'index.html')


@app.route('/web/<path:path>')
def send_web_files(path):
    """Serve static files from the 'web' directory."""
    return send_from_directory('web', path)


@app.route('/background/<path:path>')
def send_background_files(path):
    """Serve static files from the 'background' directory."""
    return send_from_directory('background', "bg-server.jpg")


@app.route('/capture_motion', methods=['POST'])
def capture_motion():
    """Process the motion capture data sent by the client."""
    global received_landmarks
    if received_landmarks:
        data = np.frombuffer(received_landmarks, dtype=np.float16).reshape(33, -1)
        landmarks_queue.append(data)
        logging.debug("Update transformation from received landmark")
        response = motion_handler(0, 0, [data, None, None], from_server=False)
    else:
        with open("tmp.pkl", "rb") as rf:
            landmarks = pickle.load(rf)
            logging.debug("Update transformation from file")
            response = motion_handler(0, 0, landmarks)
    return jsonify(response)


@app.route('/set_motion', methods=['POST'])
def set_motion():
    """Receive and store motion data from the UDP server."""
    global received_landmarks
    j_request = request.get_json()
    received_landmarks = base64.b64decode(j_request["ldmk"])
    logging.info("Motion data updated.")
    return "Success"


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000, debug=True)
