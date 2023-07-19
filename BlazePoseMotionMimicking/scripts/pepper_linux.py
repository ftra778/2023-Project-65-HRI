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

def main(session):

    # Get the services ALMotion & ALRobotPosture.

    names = [
                "LShoulderRoll", "LElbowRoll",
                "RShoulderRoll", "RElbowRoll",
                "HeadYaw", "HeadPitch",
                "LShoulderPitch", "RShoulderPitch",
                "LElbowYaw", "RElbowYaw", "HipRoll", "HipPitch"
    ]

    name_of_save = r"celebrate"
    df = pd.read_csv(r"/afs/ec.auckland.ac.nz/users/f/t/ftra778/unixhome/Documents/videos/" + name_of_save + r".csv")
    columns = list(df.columns.values)
    JointAngles = [h for h in columns if 'Time' not in h]

    # Command angles
    HipRoll = df['HipRoll'].values.tolist()
    HipPitch = df['HipPitch'].values.tolist()
    HeadYaw = df['HeadYaw'].values.tolist()
    HeadPitch = df['HeadPitch'].values.tolist()
    LShoulderPitch = df['LShoulderPitch'].values.tolist()
    RShoulderPitch = df['RShoulderPitch'].values.tolist()
    LElbowYaw = df['LElbowYaw'].values.tolist()
    RElbowYaw = df['RElbowYaw'].values.tolist()
    LShoulderRoll = df['LShoulderRoll'].values.tolist()
    RShoulderRoll = df['RShoulderRoll'].values.tolist()
    LElbowRoll = df['LElbowRoll'].values.tolist()
    RElbowRoll = df['RElbowRoll'].values.tolist()
    T = df['Time'].values.tolist()

    movement = {'RShoulderRoll': RShoulderRoll, 'LShoulderRoll': LShoulderRoll,
                'RElbowRoll': RElbowRoll, 'LElbowRoll': LElbowRoll, 'HeadYaw': HeadYaw,
                'HeadPitch': HeadPitch, 'LShoulderPitch': LShoulderPitch,
                'RShoulderPitch': RShoulderPitch, 'LElbowYaw': LElbowYaw,
                'RElbowYaw': RElbowYaw, 'HipRoll': HipRoll, 'HipPitch': HipPitch}

    motion_service = session.service("ALMotion")
    motion_service.setStiffnesses(names, 1.0)

    names = JointAngles
    angleLists = [movement[JointAngles[i]] for i in range(len(JointAngles))]
    timeLists = [T for j in range(len(JointAngles))]
    isAbsolute  = True
    motion_service.angleInterpolation(names, angleLists, timeLists, isAbsolute)

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
