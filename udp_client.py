import socket
import time

import cv2
import mediapipe as mp
import numpy as np


class MotionClient:
    def __init__(self, video_source, motion_addr, bg_addr, resize_ratio=3):
        self.motion_addr = motion_addr
        self.bg_addr = bg_addr
        self.resize_ratio = resize_ratio
        self.bg_cache = None
        self.bg_seg_idx = 0
        self.bg_seg_length = 200
        self.udp_client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.cap = cv2.VideoCapture(video_source)

    def send_motion_data(self, keypoints):
        if keypoints is not None:
            self.udp_client_socket.sendto(keypoints, self.motion_addr)
            print(f"Sent motion data of length {len(keypoints)}")

    def send_background(self, image):
        if self.bg_cache is None:
            resize_dim = (image.shape[1] // self.resize_ratio, image.shape[0] // self.resize_ratio)
            background = cv2.resize(image, resize_dim)
            background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
            _, enc_bg = cv2.imencode('.jpg', background, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            self.bg_cache = enc_bg.tobytes()

        segment_end = (self.bg_seg_idx + 1) * self.bg_seg_length
        segment = self.bg_cache[self.bg_seg_idx * self.bg_seg_length:segment_end]

        if segment:
            self.udp_client_socket.sendto(segment, self.bg_addr)
            print(f"Sent background segment of length {len(segment)}")
            self.bg_seg_idx += 1
        else:
            # Reset cache and index when all segments have been sent
            self.bg_cache = None
            self.bg_seg_idx = 0

    def process_and_send(self):
        with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print('Restarting video')
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                if results.pose_landmarks:
                    keypoints = []
                    for data_point in results.pose_landmarks.landmark:
                        keypoints += [data_point.x, data_point.y, data_point.z, data_point.visibility]
                    keypoints = np.asarray(keypoints)
                    ldmks = np.float16(keypoints).tobytes()
                    self.send_motion_data(ldmks)

                self.send_background(image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

                time.sleep(0.2)


if __name__ == "__main__":
    video_source = "data/dance.mov"
    motion_recv_addr = ("127.0.0.1", 20003)
    bg_recv_addr = ("127.0.0.1", 20004)

    client = MotionClient(video_source, motion_recv_addr, bg_recv_addr)
    client.process_and_send()
    client.cap.release()
    cv2.destroyAllWindows()
