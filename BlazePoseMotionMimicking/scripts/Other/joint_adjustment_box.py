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
        self.timestamps = []


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
            
        fractionMaxSpeed  = 1.0
        iter_count = 10
        first_pass = True

        self.tts.say("Beginning Movement")
        #for i in range(0, iter_count):
        while(1):
            i = 0

            with open(r"D:\Uni\700\PoseTests\joint-angles.csv", mode ='r') as file:
                #reading the CSV file
                csvFile = csv.reader(file)

                for lines in csvFile:
                    if i != 0:
                        for j in range(0, lines.__len__()):
                            lines[j] = float(lines[j])
                        if first_pass is False:
                            self.angles = moveLimiter(lines, self.current_angles, 0.4, 0.01)
                        else:
                            self.angles = lines
                            first_pass = False
                    else:
                        i= i + 1

            self.motion_services.setAngles(self.names, self.angles, fractionMaxSpeed)
            time.sleep(0.05)
            self.current_angles = self.motion_services.getAngles(self.names, True)
            #time.sleep(0.5)


        self.tts.say("Movement Finished")
        time.sleep(0.5)
        self.onStopped() #activate the output of the box

        pass

    def onInput_onStop(self):
        self.onUnload() #it is recommended to reuse the clean-up as the box is stopped
        self.onStopped() #activate the output of the box
