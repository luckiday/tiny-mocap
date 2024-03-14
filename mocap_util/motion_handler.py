from collections import namedtuple

from .ik import to_trans_dict

ResultLandmarks = namedtuple("ResultLandmarks", ["pose_world_landmarks", "left_hand_landmarks", "right_hand_landmarks"])


def motion_handler(client_id, time, landmarks, from_server=True):
    """
    Convert the motion animation to kinematic transformations
    :param client_id:
    :param time:
    :param landmark:
    :return: transformation dictionary
    """
    landmarks = ResultLandmarks(*landmarks)
    trans_dict = to_trans_dict(landmarks[0], landmarks[1],
                               landmarks[2], from_server=from_server, output_format="Quaternion")
    return trans_dict
