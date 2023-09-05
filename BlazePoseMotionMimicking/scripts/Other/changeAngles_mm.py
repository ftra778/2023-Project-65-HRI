import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import math
# from datetime import datetime
import time

# MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

webcam_cap = 0

# Input the video
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

if webcam_cap == 0:
    cap = cv2.VideoCapture(r"D:\Uni\700\PoseTests\fast_arms.mp4")
else:
    cap = cv2.VideoCapture(0) # 0 for web camera input

# TimeStamp = [] # Time stamps list in real-time
prevAngles = []

stepthrough = 0
cap_fail = 0
frame_rate = 90
prev = 0

# Arms landmarks
landmark1 = 'LeftShoulder'
landmark2 = 'LeftHip'
landmark3 = 'LeftElbow'
landmark4 = 'LeftWrist'
landmark5 = 'RightShoulder'
landmark6 = 'RightHip'
landmark7 = 'RightElbow'
landmark8 = 'RightWrist'

# Loop for each frame
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    time_elapsed = time.time() - prev
    res, image = cap.read()

    if (time_elapsed > 1./frame_rate) or (webcam_cap):
        prev = time.time()

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

        # Get the current time
        # time = datetime.now()
        # TimeStamp.append(time)

        # Get the pose coordinates of each landmark
        image_height, image_width, _ = image.shape # Normalize the reference frame according to the resolution of the video.

        # Get the left shoulder coordinates
        try:
            X11 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
            Y11 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
        except AttributeError:
            X11 = 0
            Y11 = 0
            cap_fail = 1

        # Get the right shoulder coordinates
        try:
            X12 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
            Y12 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height
        except AttributeError:
            X12 = 0
            Y12 = 0
            cap_fail = 1

        # Get the left elbow coordinates
        try:
            X13 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width
            Y13 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height
        except AttributeError:
            X13 = 0
            Y13 = 0
            cap_fail = 1

        # Get the right elbow coordinates
        try:
            X14 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width
            Y14 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height
        except AttributeError:
            X14 = 0
            Y14 = 0
            cap_fail = 1

        # Get the left wrist coordinates
        try:
            X15 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width
            Y15 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height
        except AttributeError:
            X15 = 0
            Y15 = 0
            cap_fail = 1

        # Get the right wrist coordinates
        try:
            X16 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width
            Y16 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height
        except AttributeError:
            X16 = 0
            Y16 = 0
            cap_fail = 1

        # Get the left hip coordinates
        try:
            X23 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width
            Y23 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height
        except AttributeError:
            X23 = 0
            Y23 = 0
            cap_fail = 1

        # Get the right hip coordinates
        try:
            X24 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width
            Y24 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height

        except AttributeError:
            X24 = 0
            Y24 = 0
            cap_fail = 1    

        # Get the nose coordinates
        try:
            X0 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width
            Y0 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height
        except AttributeError:
            X0 = 0
            Y0 = 0
            cap_fail = 1

        if cap_fail == 1:
            cap_fail = 0
        else:


            # Joint Angles in 2D
            xy = {  'LeftShoulder': np.column_stack([X11, Y11]), 'RightShoulder': np.column_stack([X12, Y12]),
                    'LeftElbow': np.column_stack([X13, Y13]), 'RightElbow': np.column_stack([X14, Y14]),
                    'LeftWrist': np.column_stack([X15, Y15]), 'RightWrist': np.column_stack([X16, Y16]),
                    'LeftHip': np.column_stack([X23, Y23]), 'RightHip': np.column_stack([X24, Y24])}

            # Calculate each body segment vector in 2D
            LS_LE = xy[landmark3] - xy[landmark1]  # Left Shoulder to Left Elbow
            LE_LS = xy[landmark1] - xy[landmark3]  # Left Elbow to Left Shoulder
            LW_LE = xy[landmark3] - xy[landmark4]  # Left Wrist to Left Elbow
            LS_LH = xy[landmark2] - xy[landmark1]  # Left Shoulder to Left Hip

            RS_RE = xy[landmark7] - xy[landmark5]  # Right Shoulder to Right Elbow
            RE_RS = xy[landmark5] - xy[landmark7]  # Right Elbow to Right Shoulder
            RW_RE = xy[landmark7] - xy[landmark8]  # Right Wrist to Right Elbow
            RS_RH = xy[landmark6] - xy[landmark5]  # Right Shoulder to Right Hip

            RS_LS = xy[landmark1] - xy[landmark5]  # Right Shoulder to Left Shoulder


            
            # Original Lengths of arm segments
            if stepthrough == 0:
                l1_left = np.linalg.norm(LE_LS[:])  # Upper left arm
                l2_left = np.linalg.norm(LW_LE[:])  # Lower left arm

                l1_right = np.linalg.norm(RE_RS[:])  # Upper right arm
                l2_right = np.linalg.norm(RW_RE[:])  # Lower right arm

                shoulders = np.linalg.norm(RS_LS[:])   # Shouler original width
                torso = Y24 - Y12   # Torso original height
                hpa = Y12 - Y0      # Head original angle
            
            # Using Taylor's algorithm to calculate the depth between two orthographic projected points
            # Calculate depth between left shoulder and left elbow
            dz = (l1_left ** 2) - (((X11 - X13) ** 2) + ((Y11 - Y13) ** 2))
            if dz <= 0.0:
                dz = 0.0
            Z3 = math.sqrt(dz)
                

            # Calculate depth between right shoulder and right elbow
            dz = (l1_right ** 2) - (((X12 - X14) ** 2) + ((Y12 - Y14) ** 2))
            if dz <= 0.0:
                dz = 0.0
            Z4 = math.sqrt(dz)

            # Calculate depth between left elbow and left wrist
            dz = (l2_left ** 2) - (((X13 - X15) ** 2) + ((Y13 - Y15) ** 2))
            if dz <= 0.0:
                dz = 0.0
            dz2_left = math.sqrt(dz)
            Z5 = dz2_left + Z3

            # Calculate depth between right elbow and right wrist
            dz = (l2_right ** 2) - (((X14 - X16) ** 2) + ((Y14 - Y16) ** 2))
            if dz <= 0.0:
                dz = 0.0
            dz2_right = math.sqrt(dz)
            Z6 = dz2_right + Z4


            # Calculate the hip roll angles
            # Current horizontal distance between the 2 shoulders over the distance between them
            adj = (X11 - X12) / np.linalg.norm(RS_LS[:])
            if adj >= 1:          # Keeping the ratio less than or equal to 1
                adj = 1
            phi = np.arccos(adj)  # Arc cos to get the angle
            if phi >= 0.5149:     # Maximum right hip roll is 29.5°.
                phi = 0.5149
            if Y12 < Y11:     # If right shoulder is above the left shoulder then the direction of hip roll is reversed.
                phi = phi * -1
            if phi <= -0.5149:    # Maximum left hip roll is -29.5°.
                phi = -0.5149
            HipRoll = -(phi)

            # Calculate the hip pitch angles
            adj = (Y24 - Y12) / torso  # Current height of torso over the original height
            if adj >= 1:                   # Keeping the ratio less than or equal to 1
                adj = 1
            phi = np.arccos(adj)           # Hip pitch angle for leaning forward is negative
            if np.linalg.norm(RS_LS[:]) - shoulders > 0:    # If the shoulders are closer to the camera, it indicates a lean forward, and vice versa NEW
                phi = phi * (-2)                            # Double value since forward leaning detection doesn't work properly
            if phi >= 1.0385:              # Maximum hip pitch angle is 59.5°.
                phi = 1.0385
            elif phi <= -1.0385:
                phi = -1.0385

            HipPitch = phi

            # Calculate the head yaw angles
            d = np.linalg.norm(RS_LS[:]) / 2  # Half of initial distance between right and left shoulder

            if (X0 - X12) / d >= 0.9 and (X0 - X12) / d <= 1.1:  # Estimating the angle to be 0° if the nose
                hy = 0.0                                                   # X coordinate doesn't exceed 10% from each side

            elif (X0 - X12) / d < 0.9:                                # Angle of looking to the right based on how much
                hy = ((d - (X0 - X12)) / d) * -(np.pi / 2)            # the nose is approaching the right shoulder.
                if hy <= -np.pi / 2:                                       # Maximum head yaw angle to the right is -90°.
                    hy = -np.pi / 2

            elif (X0 - X12) / d > 1.1:                                # Angle of looking to the right based on how much
                hy = (((X0 - X12) - d) / d) * (np.pi / 2)             # the nose is approaching the left shoulder.
                if hy >= np.pi / 2:                                        # Maximum head yaw angle to the left is 90°.
                    hy = np.pi / 2
            HeadYaw = hy

            HeadYaw = ((hy + 0.067) - (HipRoll * 2.8))               # Offset by 0.067 seems to make head yaw more accurate

            # Prevent head yaw from moving when hip rolls too far (Currently no way of confidently determining head yaw when tilting)
            # if HipRoll > 0.05 or HipRoll < -0.05:
            #     HeadYaw = 0.0

            # Calculate the head pitch angles
            h = Y12 - Y0
            if (Y12 - Y0) / hpa >= 0.95 and (Y12 - Y0) / hpa <= 1.0:
                hp = 0.0
            elif (Y12 - Y0) / hpa < 0.95:
                hp = ((h - (Y12 - Y0)) / hpa) * 0.6371
                if hp >= 0.6371:
                    hp = 0.6371
            elif (Y12 - Y0) / hpa > 1.0:
                hp = (((Y12 - Y0) - h) / hpa) * -0.7068 * 2
                if hp <= -0.7068:
                    hp = -0.7068
            HeadPitch = hp

            # 3D coordinates
            xyz = {'LeftShoulder': np.column_stack([X11, Y11, 0]),
                'RightShoulder': np.column_stack([X12, Y12, 0]),
                'LeftElbow': np.column_stack([X13, Y13, Z3]), 'RightElbow': np.column_stack([X14, Y14, Z4]),
                'LeftWrist': np.column_stack([X15, Y15, Z5]), 'RightWrist': np.column_stack([X16, Y16, Z6])}

            # 3D vectors
            LS_LE_3D = xyz[landmark3] - xyz[landmark1]
            RS_RE_3D = xyz[landmark7] - xyz[landmark5]

            LE_LS_3D = xyz[landmark1] - xyz[landmark3]
            LW_LE_3D = xyz[landmark3] - xyz[landmark4]

            RE_RS_3D = xyz[landmark5] - xyz[landmark7]
            RW_RE_3D = xyz[landmark7] - xyz[landmark8]

            UpperArmLeft = xyz[landmark3] - xyz[landmark1]
            UpperArmRight = xyz[landmark7] - xyz[landmark5]

            ZeroXLeft = xyz[landmark3] - xyz[landmark1]
            ZeroXRight = xyz[landmark7] - xyz[landmark5]

            ZeroXLeft[0, 0] = 0
            ZeroXRight[0, 0] = 0

            UpperArmLeft[0, 1] = 0
            UpperArmRight[0, 1] = 0

            if stepthrough == 0:
                LS_LE_3D_init = LS_LE_3D
                RS_RE_3D_init = RS_RE_3D
                ZeroXLeft_init = ZeroXLeft
                ZeroXRight_init = ZeroXRight
                stepthrough = stepthrough + 1



            # Calculate the left shoulder roll angles
            temp1 = (np.dot(LS_LE_3D[0, :], ZeroXLeft[0, :])) / (np.linalg.norm(LS_LE_3D[0, :]) * np.linalg.norm(ZeroXLeft[0, :]))
            temp = np.arccos(temp1)
            if temp >= 1.56:
                temp = 1.56
            if temp <= np.arccos((np.dot(LS_LE_3D_init[0, :], ZeroXLeft_init[0, :])) / (np.linalg.norm(LS_LE_3D_init[0, :]) * np.linalg.norm(ZeroXLeft_init[0, :]))):
                temp = 0.01
            # if prevAngles:
            #     if abs(temp - prevAngles[0]) > 0.03:
            #         temp = prevAngles[0] + 0.03
            #     if abs(temp - prevAngles[0]) < -0.03:
            #         temp = prevAngles[0] - 0.03
            # if prevAngles:
            #     print(prevAngles)
            LShoulderRoll = temp

            # Calculate the right shoulder roll angles
            temp1 = (np.dot(RS_RE_3D[0, :], ZeroXRight[0, :])) / (np.linalg.norm(RS_RE_3D[0, :]) * np.linalg.norm(ZeroXRight[0, :]))
            temp = np.arccos(temp1)
            if temp >= 1.56:
                temp = -1.56
            else:
                temp = temp * (-1)
            if temp > -np.arccos((np.dot(RS_RE_3D_init[0, :], ZeroXRight_init[0, :])) / (np.linalg.norm(RS_RE_3D_init[0, :]) * np.linalg.norm(ZeroXRight_init[0, :]))):
                temp = 0.01
            RShoulderRoll = temp

            # Prevent shoulders from rolling when hip rolls 
            if (HipRoll > 0):
                RShoulderRoll = (RShoulderRoll + HipRoll)
            else:
                LShoulderRoll = (LShoulderRoll + HipRoll)

            # Calculate the left elbow roll angles
            temp1 = (np.dot(LE_LS_3D[0, :], LW_LE_3D[0, :])) / (np.linalg.norm(LE_LS_3D[0, :]) * np.linalg.norm(LW_LE_3D[0, :]))
            temp = np.arccos(temp1)
            if temp >= 1.56:
                temp = -1.56
            else:
                temp = temp * -1
            LElbowRoll = temp

            # Calculate the right elbow roll angles
            temp1 = (np.dot(RE_RS_3D[0, :], RW_RE_3D[0, :])) / (np.linalg.norm(RE_RS_3D[0, :]) * np.linalg.norm(RW_RE_3D[0, :]))
            temp = np.arccos(temp1)
            if temp >= 1.56:
                temp = 1.56
            RElbowRoll = temp

            # Calculate the left shoulder pitch & left elbow yaw angles
            temp1 = (np.dot(UpperArmLeft[0, :], LS_LE_3D[0, :])) / (np.linalg.norm(UpperArmLeft[0, :]) * np.linalg.norm(LS_LE_3D[0, :]))
            temp = np.arccos(temp1)
            if temp >= np.pi / 2:
                temp = np.pi / 2
            if Y11 > Y13:
                temp = temp * -1
            LShoulderPitch = temp

            if LShoulderRoll <= 0.4:
                ley = -np.pi / 2
            elif Y13 - Y15 > 0.2 * l2_left:
                ley = -np.pi / 2
            elif Y13 - Y15 < 0 and -(Y13 - Y15) > 0.2 * l2_left and LShoulderRoll > 0.7:
                ley = np.pi / 2
            else:
                ley = 0.0
            LElbowYaw = ley

            # Calculate the right shoulder pitch & right elbow yaw angles
            temp1 = (np.dot(UpperArmRight[0, :], RS_RE_3D[0, :])) / (np.linalg.norm(UpperArmRight[0, :]) * np.linalg.norm(RS_RE_3D[0, :]))
            temp = np.arccos(temp1)
            if temp >= np.pi / 2:
                temp = np.pi / 2
            if Y12 > Y14:
                temp = temp * -1
            RShoulderPitch = temp

            if RShoulderRoll >= -0.4:
                rey = np.pi / 2
            elif Y14 - Y16 > 0.2 * l2_right:
                rey = np.pi / 2
            elif Y14 - Y16 < 0 and -(Y14 - Y16) > 0.2 * l2_right and RShoulderRoll < -0.7:
                rey = -np.pi / 2
            else:
                rey = 0.0
            RElbowYaw = rey

            # T = []
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

            # def SafePosition(T, AngleRobot):
            #     T.append(T[-1] + 2)  # 2 seconds to reach the safe initial position

            #     for i in range(4, len(AngleRobot)):
            #         AngleRobot[i].append(0.0)

            #     for i in range(3):
            #         AngleRobot[i].append(np.pi / 2)

            #     AngleRobot[3].append(-np.pi / 2)




            # SafePosition(T, AngleRobot)
            AngleHuman = [LShoulderPitch, RShoulderPitch, RElbowYaw, LElbowYaw, LShoulderRoll, LElbowRoll,
                        RShoulderRoll, RElbowRoll, HeadYaw, HeadPitch, HipRoll, HipPitch]
            prevAngles = AngleHuman

            # Export data to input to Pepper
            angles = {'LShoulderRoll': LShoulderRoll, 'LElbowRoll': LElbowRoll,
                    'RShoulderRoll': RShoulderRoll, 'RElbowRoll': RElbowRoll, 'HeadYaw': HeadYaw,
                    'HeadPitch': HeadPitch, 'LShoulderPitch': LShoulderPitch, 'RShoulderPitch': RShoulderPitch,
                    'LElbowYaw': LElbowYaw, 'RElbowYaw': RElbowYaw, 'HipRoll': HipRoll, 'HipPitch': HipPitch}

            ThetaR = pd.DataFrame.from_dict([angles])
            ThetaR.to_csv(r"D:\Uni\700\PoseTests\joint-angles.csv", index=False)









# Close MediaPipe Pose
pose.close()
cap.release()
