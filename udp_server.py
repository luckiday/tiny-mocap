import base64
import fcntl
import os
import socket

import requests

set_motion_url = 'http://127.0.0.1:8000/set_motion'


def udp_motion_receiver():
    """
    Receive landmarks from UDP port
    :return:
    """
    localIP = "127.0.0.1"
    motionPort = 20003
    bgPort = 20004
    bufferSize = 4096

    bgSegLength = 200
    bgCache = None

    # Create UDP socket to receive motion
    UDPMotionSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPMotionSocket.bind((localIP, motionPort))
    fcntl.fcntl(UDPMotionSocket, fcntl.F_SETFL, os.O_NONBLOCK)
    print("motion_receiver up and listening")
    # Create UDP socket to receive background image
    UDPBgSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPBgSocket.bind((localIP, bgPort))
    # fcntl.fcntl(UDPBgSocket, fcntl.F_SETFL, os.O_NONBLOCK)
    print("bg_receiver up and listening")

    # Listen for incoming datagrams

    while True:
        try:
            data, addr = UDPMotionSocket.recvfrom(bufferSize)
            # with open("tmp.pkl", "wb") as output_file:
            #     output_file.write(data)
            #     print("File updated")

            # Post the latest landmark to animation server
            post_obj = {"ldmk": base64.b64encode(data).decode('ascii')}
            x = requests.post(set_motion_url, json=post_obj)
            print("server response:", x)
        except BlockingIOError:
            continue

        try:
            bgdata, bgaddr = UDPBgSocket.recvfrom(bufferSize)
            print("receive background data:", len(bgdata))
            bgFileName = "background/bg-server.jpg"
            if len(bgdata) < bgSegLength:
                bgCache += bgdata
                output_file = open(bgFileName, "wb")
                output_file.seek(0)
                output_file.write(bgCache)
                output_file.truncate()
                output_file.close()
                print("Background Updated, file length: ", len(bgCache))
                bgCache = None
            else:
                if bgCache:
                    bgCache += bgdata
                else:
                    bgCache = bgdata
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            break

        # print(f"Receive landmarks {landmarks}")
        # transdict = motion_handler(0, 0, landmarks)


if __name__ == '__main__':
    udp_motion_receiver()
