import base64
import collections
import logging
import pickle

import numpy as np
from flask import Flask, send_from_directory, jsonify, request

from mocap_util.motion_handler import motion_handler

# logging.basicConfig(filename='tmp.log', encoding='utf-8', level=logging.DEBUG)
logging.basicConfig(encoding="utf-8", level=logging.DEBUG)

app = Flask(__name__)
app.add_url_rule("/", endpoint="capture")
app.add_url_rule("/capture", endpoint="capture")
app.add_url_rule("/background", endpoint="background")

landmarks = collections.deque(maxlen=10)

received_landmarks = None


@app.endpoint("capture")
def capture():
    # Frontend
    return send_from_directory('web', 'index.html')


@app.route('/web/<path:path>')
def send_capture_util(path):
    # Frontend
    return send_from_directory('web', path)


@app.route('/background/<path:path>')
def send_background(path):
    return send_from_directory('background', 'bg-server.jpg')


@app.route('/capture_motion', methods=['POST'])
def ik_transformation():
    """
    # Backend
    Client sends the landmark and server transform to motion animation
    :return: ik transformations
    """
    global received_landmarks
    # j_request = request.get_json()
    # channel = j_request["id"]
    # frame_time = j_request["time"]
    # landmarks = j_request["ldmk"]
    # print("Requesting new motion")
    # print(f"Current buffer {len(landmarks)}")
    if received_landmarks:
        # landmarks = pickle.loads(received_landmarks)
        data = np.frombuffer(received_landmarks, dtype=np.float16)
        data = data.reshape(33, -1)
        landmarks = [data, None, None]
        # print(landmarks)
        print("Update transDict from received landmark")
        response = motion_handler(0, 0, landmarks, from_server=False)
        # print(f"Response to browser {response}")
        # logging.info("Receive request from {}.".format(channel))
        return jsonify(response)
    else:
        with open("tmp.pkl", "rb") as rf:
            landmarks = pickle.load(rf)
            print("Update transDict from file")
            response = motion_handler(0, 0, landmarks)
            # print(f"Response to browser {response}")
            # logging.info("Receive request from {}.".format(channel))
            return jsonify(response)


@app.route('/set_motion', methods=['POST'])
def set_motion():
    """
    # Backend
    UDPserver set the landmark and server transform to motion animation
    :return: ik transformations
    """
    global received_landmarks
    j_request = request.get_json()
    landmarks = j_request["ldmk"]
    received_landmarks = base64.b64decode(landmarks)
    # print(landmarks)
    # logging.info("Response to channel {}: {}".format(channel, response))
    # logging.info("Receive request from {}.".format(channel))
    return "Success"


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000, debug=True)
