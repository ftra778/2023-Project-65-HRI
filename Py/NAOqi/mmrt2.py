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
    weights = [1,5,18]
    motion_service = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")
    first_pass = True

    # tts = ALProxy('ALTextToSpeech')
    names =   [
                        "LShoulderRoll", "LElbowRoll",
                        "RShoulderRoll", "RElbowRoll",
                        "HeadYaw", "HeadPitch",
                        "LShoulderPitch", "RShoulderPitch",
                        "LElbowYaw", "RElbowYaw", "HipRoll", "HipPitch"
                    ]
    motion_settings = {
                    'beckon': ['upper', weights],
                    'beckon1': ['upper', weights],
                    'beckon2': ['upper', weights],
                    'big-wave': ['arms', weights],
                    'bow' : ['hips', weights],
                    'celebrate' : ['arms', weights],
                    'gaze' : ['upper', weights],
                    'head-shake' : ['head', weights],
                    'high-five-give' : ['arms', weights],
                    'high-five-receive' : ['arms', weights],
                    'shrug' : ['arms', weights],
                    't-pose' : ['upper', weights],
                    'wave' : ['arms', weights],
                    'wave1' : ['arms', weights],
                    'beckonr': ['upper', weights],
                    'big-waver': ['arms', weights],
                    'bowr' : ['hips', weights],
                    'celebrater' : ['arms', weights],
                    'head-shaker' : ['head', weights],
                    'high-five-giver' : ['arms', weights],
                    'high-five-receiver' : ['arms', weights],
                    'shrugr' : ['arms', weights],
                    'waver' : ['arms', weights],
                    'test' : ['arms', weights],
                    'hold-on' : ['arms', weights],
                    'demo' : ['upper', weights]
                   }

    # Joint information: {'name': [Max Speed, Lowest Angle Value, Highest Angle Value]}
    joint_information = {
                    'LShoulderRoll': [9.0, 0.0085, 1.56],
                    'LElbowRoll': [9.0, -1.56, -0.0085],
                    'RShoulderRoll': [9.0, -1.56, -0.0085],
                    'RElbowRoll': [9.0, 0.0085, 1.56],
                    'HeadYaw': [7.0, -2.06, 2.06],
                    'HeadPitch': [9.0, -0.704, 0.443],
                    'LShoulderPitch': [7.0, -2.083, 2.083],
                    'RShoulderPitch': [7.0, -2.083, 2.083],
                    'LElbowYaw': [7.0, -2.083, 2.083],
                    'RElbowYaw': [7.0,-2.083, 2.083],
                    'HipRoll': [2.0, -0.512, 0.512],
                    'HipPitch': [2.5, -0.101, 0.101]
                    }

    # Limit the movement and speed of movements 
    def move_limiter(data, name, time):
        for i in range(len(data[1:])):
            
            # If joint is moving too fast, reassign angles by an acceptable amount
            if ((data[i] > data[i-1]) and (data[i] - data[i-1])/(time[i] - time[i-1]) > (joint_information[name][0])):
                data[i] = data[i-1] + ((time[i] - time[i-1]) * (joint_information[name][0]))
            elif ((data[i] < data[i-1]) and (data[i] - data[i-1])/(time[i] - time[i-1]) < -(joint_information[name][0])):
                data[i] = data[i-1] - ((time[i] - time[i-1]) * (joint_information[name][0]))

            # If joint is outside of range, reassign to min/max range bounds
            if data[i] < joint_information[name][1]:
                data[i] = joint_information[name][1]  
            if data[i] > joint_information[name][2]:
                data[i] = joint_information[name][2]

    angles = [0.0] * len(names)
    current_angles = [0.0] * len(names)
    timestamps = []
    motion_service.setStiffnesses(names, 1.0)

    start_time, curr_time = time.time(), time.time()
    

    while((curr_time - start_time) < max_rt_s):
        i = 0
        curr_time = time.time()
        with open(r"/home/cyin631/Documents/P4P-Human-Robot-Interaction/2023-Project-65-HRI/Excel/joint-angles", mode ='r') as file:
            #reading the CSV file
            csvFile = csv.reader(file)

            for lines in csvFile:
                if i != 0:
                    for j in range(0, lines.__len__()):
                        lines[j] = float(lines[j])
                    angles = lines
                else:
                    i= i + 1
            if first_pass is False:
                motion_service.angleInterpolationWithSpeed(names, angles[1:], 0.2)

                time.sleep(0.05)
            else:
                first_pass = False
            current_angles = angles
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

