import random
import math
import json
import csv

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

        self.json_path = r"D:\openpose\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\openpose\examples\json\Body\OIP2_keypoints.json"


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

        fractionMaxSpeed  = 0.6
        self.tts.say("Beginning Movement")
        #self.motion_services.setAngles([self.names[3], self.names[9]], [self.angles[3], self.angles[9]], fractionMaxSpeed)
        #for i in range(0, 10):
        while(1):
            i = 0

            with open(r"D:\Uni\700\PoseTests\joint-angles.csv", mode ='r') as file:
                #reading the CSV file
                csvFile = csv.reader(file)

                for lines in csvFile:
                    if i != 0:
                        for j in range(0, lines.__len__()):
                            lines[j] = float(lines[j])
                        self.angles = lines

                    else:
                        i= i + 1
            self.motion_services.setAngles(self.names, self.angles, fractionMaxSpeed)

            #time.sleep(0.5)
        # TRY PAUSE ALL COMPUTATION UNTIL MOVEMENT FINISHES
        self.tts.say("Movement Finished")
        time.sleep(0.5)
        self.onStopped() #activate the output of the box

        pass

    def onInput_onStop(self):
        self.onUnload() #it is recommended to reuse the clean-up as the box is stopped
        self.onStopped() #activate the output of the box
