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

        fractionMaxSpeed  = 0.8
        self.tts.say("Beginning Movement")
        i = 0
        step = 1

        lsr = [[self.names[0]], ['T'], ['Angles'], ['Sensors'], ['No Sens']]
        ler = [[self.names[1]], ['T'], ['Angles'], ['Sensors'], ['No Sens']]
        rsr = [[self.names[2]], ['T'], ['Angles'], ['Sensors'], ['No Sens']]
        rer = [[self.names[3]], ['T'], ['Angles'], ['Sensors'], ['No Sens']]
        hya = [[self.names[4]], ['T'], ['Angles'], ['Sensors'], ['No Sens']]
        hep = [[self.names[5]], ['T'], ['Angles'], ['Sensors'], ['No Sens']]
        lsp = [[self.names[6]], ['T'], ['Angles'], ['Sensors'], ['No Sens']]
        rsp = [[self.names[7]], ['T'], ['Angles'], ['Sensors'], ['No Sens']]
        ley = [[self.names[8]], ['T'], ['Angles'], ['Sensors'], ['No Sens']]
        rey = [[self.names[9]], ['T'], ['Angles'], ['Sensors'], ['No Sens']]
        hro = [[self.names[10]], ['T'], ['Angles'], ['Sensors'], ['No Sens']]
        hpi = [[self.names[11]], ['T'], ['Angles'], ['Sensors'], ['No Sens']]

    #    lsr = [[self.names[0]], ['T'], ['Angles'], ['Sensors']]
    #    ler = [[self.names[1]], ['T'], ['Angles'], ['Sensors']]
    #    rsr = [[self.names[2]], ['T'], ['Angles'], ['Sensors']]
    #    rer = [[self.names[3]], ['T'], ['Angles'], ['Sensors']]
    #    hya = [[self.names[4]], ['T'], ['Angles'], ['Sensors']]
    #    hep = [[self.names[5]], ['T'], ['Angles'], ['Sensors']]
    #    lsp = [[self.names[6]], ['T'], ['Angles'], ['Sensors']]
    #    rsp = [[self.names[7]], ['T'], ['Angles'], ['Sensors']]
    #    ley = [[self.names[8]], ['T'], ['Angles'], ['Sensors']]
    #    rey = [[self.names[9]], ['T'], ['Angles'], ['Sensors']]
    #    hro = [[self.names[10]], ['T'], ['Angles'], ['Sensors']]
    #    hpi = [[self.names[11]], ['T'], ['Angles'], ['Sensors']]



#        for x in sorted(os.listdir(r"D:\Uni\700\PoseTests\JointEval"), key=lambda z: int(z.split('angles')[-1].split('.')[0])):
        with open((r"D:\Uni\700\PoseTests\JointEval\input-joints.csv"), mode ='r') as file:
            #reading the CSV file
            csvFile = csv.reader(file)
            i = 0

            for lines in csvFile:
                if len(lines) != 0:
                    if i != 0:
                        for j in range(0, lines.__len__()):
                            lines[j] = float(lines[j])
                        self.angles = lines



                        self.motion_services.setAngles(self.names, self.angles, fractionMaxSpeed)
                        time.sleep(0.05)

                        lsr[1].append(step)
                        ler[1].append(step)
                        rsr[1].append(step)
                        rer[1].append(step)
                        hya[1].append(step)
                        hep[1].append(step)
                        lsp[1].append(step)
                        rsp[1].append(step)
                        ley[1].append(step)
                        rey[1].append(step)
                        hro[1].append(step)
                        hpi[1].append(step)

                        lsr[2].append(self.angles[0])
                        ler[2].append(self.angles[1])
                        rsr[2].append(self.angles[2])
                        rer[2].append(self.angles[3])
                        hya[2].append(self.angles[4])
                        hep[2].append(self.angles[5])
                        lsp[2].append(self.angles[6])
                        rsp[2].append(self.angles[7])
                        ley[2].append(self.angles[8])
                        rey[2].append(self.angles[9])
                        hro[2].append(self.angles[10])
                        hpi[2].append(self.angles[11])

                        lsr[3].append(self.motion_services.getAngles(self.names[0], True)[0])
                        ler[3].append(self.motion_services.getAngles(self.names[1], True)[0])
                        rsr[3].append(self.motion_services.getAngles(self.names[2], True)[0])
                        rer[3].append(self.motion_services.getAngles(self.names[3], True)[0])
                        hya[3].append(self.motion_services.getAngles(self.names[4], True)[0])
                        hep[3].append(self.motion_services.getAngles(self.names[5], True)[0])
                        lsp[3].append(self.motion_services.getAngles(self.names[6], True)[0])
                        rsp[3].append(self.motion_services.getAngles(self.names[7], True)[0])
                        ley[3].append(self.motion_services.getAngles(self.names[8], True)[0])
                        rey[3].append(self.motion_services.getAngles(self.names[9], True)[0])
                        hro[3].append(self.motion_services.getAngles(self.names[10], True)[0])
                        hpi[3].append(self.motion_services.getAngles(self.names[11], True)[0])

                        lsr[4].append(self.motion_services.getAngles(self.names[0], False)[0])
                        ler[4].append(self.motion_services.getAngles(self.names[1], False)[0])
                        rsr[4].append(self.motion_services.getAngles(self.names[2], False)[0])
                        rer[4].append(self.motion_services.getAngles(self.names[3], False)[0])
                        hya[4].append(self.motion_services.getAngles(self.names[4], False)[0])
                        hep[4].append(self.motion_services.getAngles(self.names[5], False)[0])
                        lsp[4].append(self.motion_services.getAngles(self.names[6], False)[0])
                        rsp[4].append(self.motion_services.getAngles(self.names[7], False)[0])
                        ley[4].append(self.motion_services.getAngles(self.names[8], False)[0])
                        rey[4].append(self.motion_services.getAngles(self.names[9], False)[0])
                        hro[4].append(self.motion_services.getAngles(self.names[10], False)[0])
                        hpi[4].append(self.motion_services.getAngles(self.names[11], False)[0])

                        step = step + 1
                        ang_list = [lsr, ler, rsr, rer, hya, hep, lsp, rsp, ley, rey, hro, hpi]

                    else:
                        i= i + 1

        with open(r'D:\Uni\700\PoseTests\JointEvalOut\output-joints.csv', 'wb') as f:
            w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for ang in ang_list:
                for val in range(0, 5):
                    w.writerow(ang[val])
                w.writerow([])

        self.tts.say("Movement Finished")
        time.sleep(0.5)
        self.onStopped() #activate the output of the box

    pass

    def onInput_onStop(self):
        self.onUnload() #it is recommended to reuse the clean-up as the box is stopped
        self.onStopped() #activate the output of the box
