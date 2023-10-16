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

## USER PARAMETERS
# Name of file to read and save outputs to
name_of_save = r"paper2"

# Bilateral Filter Parameters
epsilon = 1e-6          # Value close to zero to avoid special cases with math functions
k = 3                   # Value range
sigma_t = 1.0           # Neighbouring value influence (Higher value = other values have higher influence)
sigma_g = 1.5           # Value similarity influence (Higher value = other values with higher difference have higher influence)
useblf = True           # Enable use of filter to robot movements

# Weighted Moving Average Parameters
weights = [1, 5, 18]     # Weights to apply to data
usewma = True           # Enable use of filter to robot movements

# Track faces for motions that don't require the head
targetName = "Face"
faceWidth = 0.2

# Settings for motions: {'Name' : ['Body Area', 'WMA Weights']}
useWholeBody = True
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
                    'paper' : ['arms', weights],
                    'paper2' : ['arms', weights],
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


# Bilateral Rotational Filter
def compute_w(T):
    n = len(T)
    w = [0] * n
    for i in range(n - 1):
        if abs(T[i]) < epsilon:
            w[i] = math.log(abs(T[i + 1]) + epsilon)
        else:
            w[i] = math.log(abs(T[i] ** -1 * T[i + 1]) + epsilon)
    return w

def normalize(m_values):
    return [m / sum(m_values) for m in m_values]

def euclidean_distance(n, m):
    return abs(n-m)

def bof(T):
    n = len(T)
    k_range = range(-k, k + 1)

    w_values = compute_w(T)
    T_bar = [0] * n

    for i in range(n):
        m_values = [0] * len(k_range)
        b_values = [0] * len(k_range)

        for j, r in enumerate(k_range):
            spacial_difference = math.exp(-abs(r) / (2 * sigma_t ** 2))
            intensity_difference = math.exp(-euclidean_distance(i, i + r) / (2 * sigma_g ** 2))
            m_values[j] = spacial_difference * intensity_difference
        m_values = normalize(m_values)
        
        for j, r in enumerate(k_range):
            if -k <= r < 0:
                b_values[j] = sum(m_values[j] for j in range(-k, r))
            elif 0 <= r < k:
                b_values[j] = sum(m_values[j] for j in range(r + 1, k + 1))

        T_bar[i] = T[i] * math.exp(sum(b * w for b, w in zip(b_values, w_values[i - k: i + k + 1])))

    return T_bar

# Weighted Moving Average filter
def wma(data, weights=None):
    if weights is None:
        weights = list(range(1, 3))
    
    temp_data = [0 for _ in range(data.__len__() - (weights.__len__() - 1))]
    divisor = 0

    for i in range(0, weights.__len__()):
        divisor = divisor + weights[i]

    for i in range(weights.__len__() - 1, data.__len__()):
        quotient = 0
        for j in range(i-(weights.__len__()-1), i+1):
            quotient = quotient + (data[j] * weights[j + (weights.__len__()-1)-i])
        temp_data[i - weights.__len__()+1] = quotient/divisor
        
    data[weights.__len__()-1:] = temp_data
    
    return data

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

def main(session):

    names = [
                "LShoulderRoll", "LElbowRoll",
                "RShoulderRoll", "RElbowRoll",
                "HeadYaw", "HeadPitch",
                "LShoulderPitch", "RShoulderPitch",
                "LElbowYaw", "RElbowYaw", "HipRoll", "HipPitch"
    ]
    
    df = pd.read_csv(r"/afs/ec.auckland.ac.nz/users/f/t/ftra778/unixhome/Documents/videos/NewCamera/" + name_of_save + r".csv")
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
    

    # Filter data
    if useblf is True:
        for jname in [HipRoll, HipPitch, HeadYaw, HeadPitch, LShoulderPitch, RShoulderPitch, LElbowYaw, RElbowYaw, LShoulderRoll, RShoulderRoll, LElbowRoll, RElbowRoll]:
            jname = bof(jname)

    if usewma is True:
        for jname in [HipRoll, HipPitch, HeadYaw, HeadPitch, LShoulderPitch, RShoulderPitch, LElbowYaw, RElbowYaw, LShoulderRoll, RShoulderRoll, LElbowRoll, RElbowRoll]:
            jname = wma(jname, motion_settings[name_of_save][1])
    T = df['Time'].values.tolist()
    
    # Move limiter prevents motors from moving beyond max speeds
    for jname, jstring in zip([HipRoll, HipPitch, HeadYaw, HeadPitch, LShoulderPitch, RShoulderPitch, LElbowYaw, RElbowYaw, LShoulderRoll, RShoulderRoll, LElbowRoll, RElbowRoll], ['HipRoll', 'HipPitch', 'HeadYaw', 'HeadPitch', 'LShoulderPitch', 'RShoulderPitch', 'LElbowYaw', 'RElbowYaw', 'LShoulderRoll', 'RShoulderRoll', 'LElbowRoll', 'RElbowRoll']):
        jname = move_limiter(jname, jstring, T)

    movement = {'RShoulderRoll': RShoulderRoll, 'LShoulderRoll': LShoulderRoll,
                'RElbowRoll': RElbowRoll, 'LElbowRoll': LElbowRoll, 'HeadYaw': HeadYaw,
                'HeadPitch': HeadPitch, 'LShoulderPitch': LShoulderPitch,
                'RShoulderPitch': RShoulderPitch, 'LElbowYaw': LElbowYaw,
                'RElbowYaw': RElbowYaw, 'HipRoll': HipRoll, 'HipPitch': HipPitch}

    motion_service = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")
    tracker_service = session.service("ALTracker")
    tts = session.service("ALTextToSpeech")
    tracker_service.registerTarget(targetName, faceWidth)
    # tts.say("Which bombaclart dog i am?")

    # Start tracker
    tracker_service.track(targetName)
    motion_service.setStiffnesses(names, 1.0)

    # Separate data
    names = JointAngles
    angleLists = [movement[JointAngles[i]] for i in range(len(JointAngles))]
    timeLists = [T for j in range(len(JointAngles))]
    isAbsolute  = True

    # Hand settings based on motion type
    if name_of_save in ('hold-on', 'beckon', 'beckonr', 'big-wave', 'big-waver', 'high-five-give', 'high-five-giver', 'high-five-receive', 'high-five-receiver', 'shrug', 'shrugr', 'wave', 'waver'):
        motion_service.openHand('RHand')
        motion_service.openHand('LHand')
    if name_of_save in ('big-wave', 'shrug'):
        motion_service.openHand('LHand')
    if name_of_save in (['celebrate']):
        motion_service.closeHand('RHand')
    if name_of_save in (['celebrate']):
        motion_service.closeHand('LHand')
    
    # Perform movement
    if useWholeBody is True:
        motion_service.angleInterpolation(names, angleLists, timeLists, isAbsolute)
    elif motion_settings[name_of_save][0] is 'upper':
        motion_service.angleInterpolation((names[:4] + names[6:12]), (angleLists[:4] + angleLists[6:12]), (timeLists[:4] + timeLists[6:12]), isAbsolute)
    elif motion_settings[name_of_save][0] is 'arms':
        motion_service.angleInterpolation((names[:4] + names[6:10]), (angleLists[:4] + angleLists[6:10]), (timeLists[:4] + timeLists[6:10]), isAbsolute)
    elif motion_settings[name_of_save][0] is 'head':
        motion_service.angleInterpolation(names[4:6], angleLists[4:6], timeLists[4:6], isAbsolute)
    elif motion_settings[name_of_save][0] is 'hips':
        motion_service.angleInterpolation(names[11:12], angleLists[11:12], timeLists[11:12] , isAbsolute)
    posture_service.goToPosture('StandInit', 0.5)
    # motion_service.rest()

    # Stop tracker
    tracker_service.stopTracker()
    tracker_service.unregisterAllTargets()
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