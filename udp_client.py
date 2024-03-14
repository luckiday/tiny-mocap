import socket
import time

import cv2
import mediapipe as mp
import numpy as np

motionRecvAddr = ("127.0.0.1", 20003)
bgRecvAddr = ("127.0.0.1", 20004)

bufferSize = 4096
# Create a UDP socket at client side

UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# For webcam input:
VIDEO_SOURCE = "data/dance.mov"
cap = cv2.VideoCapture(VIDEO_SOURCE)
frame_counter = 0

bgCache = None
bgSegIdx = 0
bgSegLength = 200
resize_ratio = 3   # The compression ratio of the background image

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            if type(VIDEO_SOURCE) == int:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            else:
                print('Loop Video')
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # ---------- verison 1: send all landmark -----------
        # ldmks = [results.pose_landmarks, results.left_hand_landmarks, results.right_hand_landmarks]
        # # Send to server using created UDP socket
        # # msgFromClient = "Hello UDP Server"
        # # bytesToSend = str.encode(msgFromClient)
        # ldmks = pickle.dumps(ldmks)

        # ---------- verison 2: send compressed pose landmark -----------
        keypoints = []

        if results.pose_landmarks is None:
            continue

        for data_point in results.pose_landmarks.landmark:
            keypoints += [data_point.x, data_point.y, data_point.z, data_point.visibility]
        keypoints = np.asarray(keypoints)
        ldmks = np.float16(keypoints).tobytes()

        UDPClientSocket.sendto(ldmks, motionRecvAddr)
        print(f"Send message of length {len(ldmks)}")

        if bgCache is None:
            print(image.shape)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resize_dim = (image.shape[1]//resize_ratio, image.shape[0]//resize_ratio)
            background = cv2.resize(image, resize_dim)
            background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            result, enc_bg = cv2.imencode('.jpg', background, encode_param)
            cv2.imshow("bg.jpg", background)
            bgCache = enc_bg
        else:
            if bgSegIdx * bgSegLength < len(bgCache):
                UDPClientSocket.sendto(bgCache[bgSegIdx * bgSegLength: (bgSegIdx + 1) * bgSegLength], bgRecvAddr)
                print(f"Send background of length {len(enc_bg[bgSegIdx * bgSegLength: (bgSegIdx + 1) * bgSegLength])}")
                bgSegIdx += 1
            else:
                bgCache = None
                bgSegIdx = 0

        # msgFromServer = UDPClientSocket.recvfrom(bufferSize)
        # msg = "Message from Server {}".format(msgFromServer[0])
        # print(msg)

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

        time.sleep(0.2)
cap.release()
