import qi
import naoqi
import argparse
import sys
import random
import math
import json
import csv
import time
import pandas as pd
import numpy as np

def main(session):
    
    max_rt_s = 20
    motion_service = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")

    # tts = ALProxy('ALTextToSpeech')
    names =   [
                        "LShoulderRoll", "LElbowRoll",
                        "RShoulderRoll", "RElbowRoll",
                        "HeadYaw", "HeadPitch",
                        "LShoulderPitch", "RShoulderPitch",
                        "LElbowYaw", "RElbowYaw", "HipRoll", "HipPitch"
                    ]
    angles = [0.0] * len(names)
    current_angles = [0.0] * len(names)
    timestamps = []
    motion_service.setStiffnesses(names, 1.0)

    start_time, curr_time = time.time(), time.time()
    

    while((curr_time - start_time) < max_rt_s):
        i = 0
        curr_time = time.time()
        with open(r"/afs/ec.auckland.ac.nz/users/f/t/ftra778/unixhome/Documents/videos/joint-angles", mode ='r') as file:
            #reading the CSV file
            csvFile = csv.reader(file)

            for lines in csvFile:
                if i != 0:
                    for j in range(0, lines.__len__()):
                        lines[j] = float(lines[j])
                    angles = lines
                else:
                    i= i + 1

            motion_service.angleInterpolationWithSpeed(names, angles, 0.7)
            time.sleep(0.05)
            current_angles = motion_service.getAngles(names, True)
            #time.sleep(0.5)
    
    posture_service.goToPosture('StandInit', 0.5)
    motion_service.setStiffnesses(names, 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.22.1.21",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session)

