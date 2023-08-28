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
    
    angles = [0.0] * len(names)
    ang_list = []
    for i in names:
        ang_list.append([[i], ['T'], ['Angles'], ['Sensors']])
    fractionMaxSpeed = 0.22
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

    with open((r"head-shake.csv"), mode ='r') as file:     # NEED TO CHANGE TO INTERACT WITH PEPPER MEMORY
        #reading the CSV file
        csvFile = csv.reader(file)
        i = 0

        for lines in csvFile:
            if len(lines) != 0:
                if i != 0:
                    for j in range(0, lines.__len__()):
                        lines[j] = float(lines[j])
                    if first_pass is False:
                        angles = moveLimiter(lines, current_angles, 0.4, 0.01)
                    else:
                        angles = lines
                        first_pass = False



                    motion_service.setAngles(names[:5] + names[5 + 1:], angles[:5] + angles[5 + 1:], fractionMaxSpeed) #- line 5
                    time.sleep(0.05)

                    current_angles = motion_service.getAngles(names, True)

                    for joint in ang_list:
                        joint[1].append(step)

                    for joint, x in zip(ang_list, range(0, ang_list.__len__())):
                        joint[2].append(angles[x])

                    for next_joint, curr_joint in zip(ang_list, current_angles):
                        next_joint[3].append(curr_joint)

                    step = step + 1

                else:
                    i= i + 1

    with open(r'D:\Uni\700\PoseTests\JointEvalOut\output-joints.csv', 'wb') as f:
        w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for ang in ang_list:
            for val in range(0, 4):
                w.writerow(ang[val])
            w.writerow([])

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