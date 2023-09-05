import qi
import naoqi
import argparse
import sys
import random
import math
import json
import csv
import time

# import naoqi
# from naoqi import ALProxy
# ip = "172.22.1.21"  # Replace with Pepper's IP address
# port = 9559  # Default port for the Autonomous Life service
# life_proxy = ALProxy("ALAutonomousLife", ip, port)
# life_proxy.setState("disabled")
# life_proxy.setState("solitary")


def main(session):

    # Get the services ALMotion & ALRobotPosture.

    names = [
                "LShoulderRoll", "LElbowRoll",
                "RShoulderRoll", "RElbowRoll",
                "HeadYaw", "HeadPitch",
                "LShoulderPitch", "RShoulderPitch",
                "LElbowYaw", "RElbowYaw", "HipRoll", "HipPitch"
            ]
    
    T = []
    LSP = []
    RSP = []
    REY = []
    LEY = []
    LSR = []
    LER = []
    RSR = []
    RER = []
    HY = []
    HP = []
    HPP = []
    HPR = []
    
    JointAngles = []

    angles = [0.0] * len(names)
    ang_list = []
    for i in names:
        ang_list.append([[i], ['T'], ['Angles'], ['Sensors']])
    fractionMaxSpeed = 0.3
    i = 0
    step = 1
    motion_service = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")
    motion_service.setStiffnesses(names, 1.0)
    first_pass = True


    # Function limits the distance a joint can travel between adjacent frames
    def moveLimiter(new, curr, uppB, lowB):
        for x in range(0, 12):
            if abs(new[x] - curr[x]) > lowB:            #NEW
                if abs(new[x] - curr[x]) > uppB:
                    if (new[x] > curr[x]):
                        curr[x] = curr[x] + uppB
                    else:
                        curr[x] = curr[x] - uppB
                else:
                    curr[x] = new[x]
        return curr

    with open((r"celebrate.csv"), mode ='r') as file:     # NEED TO CHANGE TO INTERACT WITH PEPPER MEMORY
        csvFile = csv.reader(file)
        for lines in csvFile:
            if i != 0:
                for j in range(0, lines.__len__()):
                    lines[j] = float(lines[j])
            else:
                JointAngles.append(lines[1:])
                i = i+1

            # for i in range(0, names.__len__()):
            #     hro.append(lines[i])
            T.append(lines[0])
            LSP.append(lines[1])
            RSP.append(lines[2])
            REY.append(lines[3])
            LEY.append(lines[4])
            LSR.append(lines[5])
            LER.append(lines[6])
            RSR.append(lines[7])
            RER.append(lines[8])
            HY.append(lines[9])
            HP.append(lines[10])
            HPP.append(lines[11])
            HPR.append(lines[12])
    
    movement = {'RShoulderRoll': RSR, 'LShoulderRoll': LSR,
            'RElbowRoll': RER, 'LElbowRoll': LER, 'HeadYaw': HY,
            'HeadPitch': HP, 'LShoulderPitch': LSP,
            'RShoulderPitch': RSP, 'LElbowYaw': LEY,
            'RElbowYaw': REY, 'HipRoll': HPR, 'HipPitch': HPP}

    names  = JointAngles
    angleLists = [movement[JointAngles[i]] for i in range(len(JointAngles))]
    timeLists = [T for j in range(len(JointAngles))]
    isAbsolute  = True
    motion_service.angleInterpolation(names, angleLists, timeLists, isAbsolute)
    current_angles = motion_service.getAngles(names, True)

    for joint in ang_list:
        joint[1].append(step)

    for joint, x in zip(ang_list, range(0, ang_list.__len__())):
        joint[2].append(angles[x])

    for next_joint, curr_joint in zip(ang_list, current_angles):
        next_joint[3].append(curr_joint)

    step = step + 1

    motion_service.setStiffnesses(names, 1.0)
# else:
#     i= i + 1

# with open(r'D:\Uni\700\PoseTests\JointEvalOut\output-joints.csv', 'wb') as f:
#     w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     for ang in ang_list:
#         for val in range(0, 4):
#             w.writerow(ang[val])
#         w.writerow([])

    # motion_service.closeHand('LHand')

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
