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

epsilon = 1e-6
k = 3
sigma_t = 1.0
sigma_g = 1.5
weights = [1, 2, 4]

name_of_save = r"wave"

# Bilateral Human Motion Filter
def compute_w(T):
    n = len(T)
    w = [0] * n
    for i in range(n - 1):
        q_i, q_i_plus_1 = T[i], T[i + 1]
        if abs(q_i) < epsilon:
            w[i] = math.log(abs(q_i_plus_1) + epsilon)
        else:
            w[i] = math.log(abs(q_i ** -1 * q_i_plus_1) + epsilon)
    return w

def normalize(m_values):
    total_sum = sum(m_values)
    return [m / total_sum for m in m_values]

def D_fun(n, m):
    return abs(n-m)

def bof(T):
    n = len(T)
    k_range = range(-k, k + 1)

    w_values = compute_w(T)
    T_bar = [0] * n

    for i in range(n):
        m_values = [0] * len(k_range)
        for j, r in enumerate(k_range):
            m_values[j] = math.exp(-abs(r) / (2 * sigma_t ** 2)) * math.exp(-D_fun(i, i + r) / (2 * sigma_g ** 2))

        m_values = normalize(m_values)
        
        b_values = [0] * len(k_range)
        for j, r in enumerate(k_range):
            if -k <= r < 0:
                b_values[j] = sum(m_values[j] for j in range(-k, r))
            elif 0 <= r < k:
                b_values[j] = sum(m_values[j] for j in range(r + 1, k + 1))

        T_bar[i] = T[i] * math.exp(sum(b * w for b, w in zip(b_values, w_values[i - k: i + k + 1])))

    return T_bar

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


def main(session):

    # Get the services ALMotion & ALRobotPosture.

    useWMA = True

    names = [
                "LShoulderRoll", "LElbowRoll",
                "RShoulderRoll", "RElbowRoll",
                "HeadYaw", "HeadPitch",
                "LShoulderPitch", "RShoulderPitch",
                "LElbowYaw", "RElbowYaw", "HipRoll", "HipPitch"
    ]

    csvOut = []

    for i in names:
        csvOut.append([[i], ['T'], ['Angles'], ['Sensors'], ['WMA&BOF']])

    
    save_loc = r"/afs/ec.auckland.ac.nz/users/f/t/ftra778/unixhome/Documents/videos/Videos/"  + name_of_save
    df = pd.read_csv(save_loc + r".csv")
    columns = list(df.columns.values)
    JointAngles = [h for h in columns if 'Time' not in h]

    # Original angles
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

    names = JointAngles
    angleListsog = [movement[JointAngles[i]] for i in range(len(JointAngles))]
    timeLists = [T for j in range(len(JointAngles))]
    angleArray = np.array(angleListsog)
    angleArray = np.transpose(angleArray)
    angleListsog = np.ndarray.tolist(angleArray)

    for i in range(0, timeLists[0].__len__()):

        for joint in csvOut:
            joint[1].append(timeLists[0][i])

        for joint, x in zip(csvOut, range(0, csvOut.__len__())):
            joint[2].append(angleListsog[i][x])


    # Command angles
    HipRoll = wma(bof(df['HipRoll'].values.tolist()))
    HipPitch = wma(bof(df['HipPitch'].values.tolist()))
    HeadYaw = wma(bof(df['HeadYaw'].values.tolist()))
    HeadPitch = wma(bof(df['HeadPitch'].values.tolist()))
    LShoulderPitch = wma(bof(df['LShoulderPitch'].values.tolist()))
    RShoulderPitch = wma(bof(df['RShoulderPitch'].values.tolist()))
    LElbowYaw = wma(bof(df['LElbowYaw'].values.tolist()))
    RElbowYaw = wma(bof(df['RElbowYaw'].values.tolist()))
    LShoulderRoll = wma(bof(df['LShoulderRoll'].values.tolist()))
    RShoulderRoll = wma(bof(df['RShoulderRoll'].values.tolist()))
    LElbowRoll = wma(bof(df['LElbowRoll'].values.tolist()))
    RElbowRoll = wma(bof(df['RElbowRoll'].values.tolist()))

    movement = {'RShoulderRoll': RShoulderRoll, 'LShoulderRoll': LShoulderRoll,
                'RElbowRoll': RElbowRoll, 'LElbowRoll': LElbowRoll, 'HeadYaw': HeadYaw,
                'HeadPitch': HeadPitch, 'LShoulderPitch': LShoulderPitch,
                'RShoulderPitch': RShoulderPitch, 'LElbowYaw': LElbowYaw,
                'RElbowYaw': RElbowYaw, 'HipRoll': HipRoll, 'HipPitch': HipPitch}

    motion_service = session.service("ALMotion")
    motion_service.setStiffnesses(names, 1.0)
    names = JointAngles
    angleListswma = [movement[JointAngles[i]] for i in range(len(JointAngles))]
    timeLists = [T for j in range(len(JointAngles))]

    # Times for analysis
    start_time = time.time()
    curr_time = time.time()


    angleArray = np.array(angleListswma)
    angleArray = np.transpose(angleArray)
    angleListswma = np.ndarray.tolist(angleArray)
    csvAngles = []


    for i in range(0, timeLists[0].__len__()):
        # while curr_time - start_time < timeLists[0][i]:
        #     curr_time = time.time()
        
        if useWMA is True:
            # motion_service.setAngles(names, angleListswma[i], 0.3)
            motion_service.angleInterpolationWithSpeed(names, angleListswma[i], 0.6)
        else:
            motion_service.setAngles(names, angleListsog[i], 0.3)
            
        csvAngles.append(motion_service.getAngles(names, True))
        # motion_service.angleInterpolationWithSpeed(names, testvar2[i], 0.2)

    motion_service.setStiffnesses(names, 0.0)


    for i in range(0, timeLists[0].__len__()):

        for next_joint, curr_joint in zip(csvOut, csvAngles[i]):
            next_joint[3].append(curr_joint)

        for joint, x in zip(csvOut, range(0, csvOut.__len__())):
            joint[4].append(angleListswma[i][x])


    with open(save_loc + "_out.csv", 'wb') as f:
      w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      for ang in csvOut:
        for val in range(0, 5):
            w.writerow(ang[val])
        w.writerow([])


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