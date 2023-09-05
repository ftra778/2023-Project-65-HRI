import random
import math
import json
import csv
import time

class MyClass(GeneratedClass):
    def __init__(self):
        GeneratedClass.__init__(self)
        self.motion_services = ALProxy('ALMotion')
        self.tts = ALProxy('ALTextToSpeech')
        self.names =   [
                            "LShoulderRoll", "LElbowRoll",
                            "RShoulderRoll", "RElbowRoll",
                            "HeadYaw", "HeadPitch",
                            "LShoulderPitch", "RShoulderPitch",
                            "LElbowYaw", "RElbowYaw", "HipRoll", "HipPitch"
                        ]
        self.angles = [0.0] * len(self.names)
        self.current_angles = [0.0] * len(self.names)
        self.ang_list = []
        for i in self.names:
            self.ang_list.append([[i], ['T'], ['Angles'], ['Sensors']])

    def onLoad(self):
        #put initialization code here
        pass

    def onUnload(self):
        #put clean-up code here
        pass

    def onInput_onStart(self):
        # TERMS:
        #        - Roll: Rotation around x axis
        #        - Pitch: Rotation around y axis
        #        - Yaw: Rotation around z axis

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
        fractionMaxSpeed  = 0.6
        self.tts.say("Beginning Movement")
        first_pass = True
        i = 0
        step = 1
        video_path = r"D:\Uni\700\PoseTests\JointEval\input-joints.csv"

    #    self.motion_services.setStiffnesses(self.names, 0)

#        for x in sorted(os.listdir(r"D:\Uni\700\PoseTests\JointEval"), key=lambda z: int(z.split('angles')[-1].split('.')[0])):
        with open((video_path), mode ='r') as file:
            #reading the CSV file
            csvFile = csv.reader(file)
            i = 0

            for lines in csvFile:
                if len(lines) != 0:
                    if i != 0:
                        for j in range(0, lines.__len__()):
                            lines[j] = float(lines[j])
                        if first_pass is False:
                            self.angles = moveLimiter(lines, self.current_angles, 0.4, 0.01)
                        else:
                            self.angles = lines
                            first_pass = False


                        self.motion_services.setAngles(self.names, self.angles, fractionMaxSpeed)
                        time.sleep(0.05)
                        self.current_angles = self.motion_services.getAngles(self.names, True)

                        for joint in self.ang_list:
                            joint[1].append(step)

                        for joint, x in zip(self.ang_list, range(0, self.ang_list.__len__())):
                            joint[2].append(self.angles[x])

                        for next_joint, curr_joint in zip(self.ang_list, self.current_angles):
                            next_joint[3].append(curr_joint)

                        step = step + 1

                    else:
                        i= i + 1

        with open(r'D:\Uni\700\PoseTests\JointEvalOut\output-joints.csv', 'wb') as f:
            w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for ang in self.ang_list:
                for val in range(0, 4):
                    w.writerow(ang[val])
                w.writerow([])

        self.tts.say("Movement Finished")
        time.sleep(0.5)
        self.onStopped() #activate the output of the box

    pass

    def onInput_onStop(self):
        self.onUnload() #it is recommended to reuse the clean-up as the box is stopped
        self.onStopped() #activate the output of the box
