import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import math
import time
import csv



# MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

name_of_save = r'seagull'

webcam_cap = 0
sample_size = 1
bof_en = True
k = 3
sigma_t = 1.0
sigma_g = 1.5
epsilon = 1e-6
weights = [1, 2, 4]

# Input the video
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

if webcam_cap == 0:
    cap = cv2.VideoCapture(r"D:\Uni\700\PoseTests\videos\Presentable\\" + name_of_save + r".mp4")
else:
    cap = cv2.VideoCapture(0) # 0 for web camera input
success = cap.set(cv2.CAP_PROP_FPS, 2)
run_time_sec = 200.0

# save_loc = r"D:\Uni\700\PoseTests\JointEval\input-joints.csv"
save_loc = r"D:\Uni\700\PoseTests\Excel\\" + name_of_save + r".csv"






TimeStamp = [] # Time stamps list in real-time

# STATIC VARS
sample = 1                      # Integer tracks amount of samples taken
init_step = True                # Boolean tracks if frame is the initial frame for base joint coordinates
cap_fail = False                # Boolean set true if essential joints aren't initially present
curr_iter = 0                   # Integer tracks current frame
curr_sample = 0                 # Integer tracks current sample (EXCLUSIVELY FOR AVERAGING)
frame_rate = 120                # Integer sets framerate of video playback for cv2 GUI
prev = 0                        # 

# Times for analysis
start_time = time.time()
curr_time = time.time()

# Arms landmarks
landmark1 = 'LeftShoulder'
landmark2 = 'LeftHip'
landmark3 = 'LeftElbow'
landmark4 = 'LeftWrist'
landmark5 = 'RightShoulder'
landmark6 = 'RightHip'
landmark7 = 'RightElbow'
landmark8 = 'RightWrist'

#AngleHuman = [['LShoulderRoll', 'LElbowRoll', 'RShoulderRoll', 'RElbowRoll', 'HeadYaw', 'HeadPitch', 'LShoulderPitch', 'RShoulderPitch', 'LElbowYaw', 'RElbowYaw', 'HipRoll', 'HipPitch']]
AngleHuman = []

# Averaged angles to be exported to Pepper
filteredAngles = [['Time', 'LShoulderRoll', 'LElbowRoll', 'RShoulderRoll', 'RElbowRoll', 'HeadYaw', 'HeadPitch', 'LShoulderPitch', 'RShoulderPitch', 'LElbowYaw', 'RElbowYaw', 'HipRoll', 'HipPitch']]

# For mapping 3D coordinates
out_3d = [[['LeftShoulder'], ['RightShoulder'], ['LeftElbow'], ['RightElbow'], ['LeftWrist'], ['RightWrist'], ['LeftHip'], ['RightHip'], 
           ['LeftShoulderF'], ['RightShoulderF'], ['LeftElbowF'], ['RightElbowF'], ['LeftWristF'], ['RightWristF'], ['LeftHipF'], ['RightHipF']]]

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


# Loop for each frame
while (cap.isOpened()) : # and (curr_time - start_time < run_time_sec))
    success, image = cap.read()
    if not success:
        break

    
    
    time_elapsed = time.time() - prev
    res, image = cap.read()

    #if (time_elapsed > 1./frame_rate) or (webcam_cap):

    if type(image) == type(None):
        break

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
    #time = datetime.now()
    #TimeStamp.append(time)

    # Get the pose coordinates of each landmark
    image_height, image_width, _ = image.shape # Normalize the reference frame according to the resolution of the video.


    # Get the left shoulder coordinates
    try:
        X11 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
        Y11 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
    except AttributeError:
        X11 = 0
        Y11 = 0
        cap_fail = True

    # Get the right shoulder coordinates
    try:
        X12 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
        Y12 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height
    except AttributeError:
        X12 = 0
        Y12 = 0
        cap_fail = True

    # Get the left elbow coordinates
    try:
        X13 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width
        Y13 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height
    except AttributeError:
        X13 = 0
        Y13 = 0
        cap_fail = True

    # Get the right elbow coordinates
    try:
        X14 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width
        Y14 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height
    except AttributeError:
        X14 = 0
        Y14 = 0
        cap_fail = True

    # Get the left wrist coordinates
    try:
        X15 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width
        Y15 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height
    except AttributeError:
        X15 = 0
        Y15 = 0
        cap_fail = True

    # Get the right wrist coordinates
    try:
        X16 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width
        Y16 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height
    except AttributeError:
        X16 = 0
        Y16 = 0
        cap_fail = True

    # Get the left hip coordinates
    try:
        X23 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width
        Y23 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height
    except AttributeError:
        X23 = 0
        Y23 = 0
        cap_fail = True

    # Get the right hip coordinates
    try:
        X24 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width
        Y24 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height

    except AttributeError:
        X24 = 0
        Y24 = 0
        cap_fail = True

    # Get the nose coordinates
    try:
        X0 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width
        Y0 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height
    except AttributeError:
        X0 = 0
        Y0 = 0
        cap_fail = True

    # If any essentail keypoints have failed to be identified, try again
    if cap_fail == True:
        cap_fail = False
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
        
        if init_step is True:
            l1_left = np.linalg.norm(LE_LS[:])  # Upper left arm
            l2_left = np.linalg.norm(LW_LE[:])  # Lower left arm

            l1_right = np.linalg.norm(RE_RS[:])  # Upper right arm
            l2_right = np.linalg.norm(RW_RE[:])  # Lower right arm

            shoulders = np.linalg.norm(RS_LS[:])   # Shouler original width
            torso = Y24 - Y12   # Torso original height
            hpa = Y12 - Y0      # Head original angle
        
        # Function limits the distance a joint can travel between adjacent frames
        def moveLimiter(new, curr, degree):
            if abs(new - curr) > degree:
                if (new > curr):
                    curr = curr + degree
                else:
                    curr = curr - degree
            else:
                curr = new
            return curr


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

        # Save 3D angles for graphs [Shoulder & Hip coordinates are anchored in the Z axis]
        out_3d.append([ [X11, X12, X13, X14, X15, X16, X23, X24], 
                        [Y11, Y12, Y13, Y14, Y15, Y16, Y23, Y24], 
                        [0, 0, Z3, Z4, Z5, Z6, 0, 0]
                        ])


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
            phi = phi * (-2.5)
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
        HeadYaw = ((hy + 0.067) - (HipRoll * 2.8))               # Offset by 0.067 seems to make head yaw more accurate

        # Prevent head yaw from moving when hip rolls too far
        # if HipRoll > 0.05 or HipRoll < -0.05:
        #     HeadYaw = 0.0
        # if HipRoll > 0:



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

        if init_step is True:
            LS_LE_3D_init = LS_LE_3D
            RS_RE_3D_init = RS_RE_3D
            ZeroXLeft_init = ZeroXLeft
            ZeroXRight_init = ZeroXRight
            # init_step = False



        # Calculate the left shoulder roll angles
        temp1 = (np.dot(LS_LE_3D[0, :], ZeroXLeft[0, :])) / (np.linalg.norm(LS_LE_3D[0, :]) * np.linalg.norm(ZeroXLeft[0, :]))
        temp = np.arccos(temp1)
        if temp >= 1.56:
            temp = 1.56
        if temp <= np.arccos((np.dot(LS_LE_3D_init[0, :], ZeroXLeft_init[0, :])) / (np.linalg.norm(LS_LE_3D_init[0, :]) * np.linalg.norm(ZeroXLeft_init[0, :]))):
            temp = 0.01
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

        curr_time = time.time()

        AngleHuman.append([(curr_time - start_time), (LShoulderRoll), (LElbowRoll), (RShoulderRoll), (RElbowRoll), (HeadYaw), (HeadPitch), (LShoulderPitch), (RShoulderPitch), (LElbowYaw), (RElbowYaw), (HipRoll), (HipPitch)])

        curr_sample = curr_sample + 1
        curr_iter = curr_iter + 1
        # curr_time = time.time()

        # if curr_sample >= sample_size:
        #     curr_sample = 0
        #     for i in range(0, sample_size):
        #         #print(AngleHuman)
        #         T.append(AngleHuman[i][0])
        #         LSR.append(AngleHuman[i][1])
        #         LER.append(AngleHuman[i][2])
        #         RSR.append(AngleHuman[i][3])
        #         RER.append(AngleHuman[i][4])
        #         HY.append(AngleHuman[i][5])
        #         HP.append(AngleHuman[i][6])
        #         LSP.append(AngleHuman[i][7])
        #         RSP.append(AngleHuman[i][8])
        #         LEY.append(AngleHuman[i][9])
        #         REY.append(AngleHuman[i][10])
        #         HPR.append(AngleHuman[i][11])
        #         HPP.append(AngleHuman[i][12])
                    
        #     filteredAngles.append([ T[sample_size-1],
        #                             np.mean(LSR), np.mean(LER), np.mean(RSR), np.mean(RER), 
        #                             np.mean(HY), np.mean(HP), 
        #                             np.mean(LSP), np.mean(RSP), np.mean(LEY), np.mean(REY), 
        #                             np.mean(HPR), np.mean(HPP)])
            
        #    AngleHuman = []

        if init_step is True:
            init_step = False

def moving_average(angles, n=3):
    ret = []
    ret.append(angles[0])
    for i in range(1, angles.__len__()):
        ret.append(np.cumsum(angles[i], dtype=float))
        ret[i][n:] = ret[i][n:] - ret[i][:-n]
        ret[i][n - 1:] / n
    return ret

def wma(data, weights=None):
    if weights is None:
        weights = list(range(1, 5))
    
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


# Writing to CSV output
# AngleHumanBLF = []
# AngleHumanWMA = []

if bof_en is True:
    AngleHumanBLF = np.array(AngleHuman).T
    AngleHumanWMA = np.array(AngleHuman).T
    AngleHumanWMABOF = np.array(AngleHuman).T

    for j in range(len(AngleHumanBLF)):
        AngleHumanBLF[j] = bof(AngleHumanBLF[j])
        AngleHumanWMA[j] = wma(AngleHumanWMA[j])
        AngleHumanWMABOF[j] = wma(bof(AngleHumanWMA[j]))
    
    AngleHumanBLF = np.ndarray.tolist(AngleHumanBLF.T)
    AngleHumanWMA = np.ndarray.tolist(AngleHumanWMA.T)
    AngleHumanWMABOF = np.ndarray.tolist(AngleHumanWMABOF.T)




with open(save_loc, 'w', newline='') as f:
    w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for i in AngleHuman:
        w.writerow(i)
    w.writerow('')
    for i in AngleHumanBLF:
        w.writerow(i)
    w.writerow('')
    for i in AngleHumanWMA:
        w.writerow(i)
    w.writerow('')
    for i in AngleHumanWMABOF:
        w.writerow(i)

x = []
y = []
z = []
with open(r"D:\Uni\700\PoseTests\Excel\\" + name_of_save + r"coordinates.csv", 'w', newline='') as f:
    w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    print(len(out_3d))
    for i in range(len(out_3d[0])):
        w.writerow(out_3d[0][i])
        x = ['x']
        y = ['y']
        z = ['z']
        for j in range(len(out_3d)-1):
            x.append(out_3d[j+1][0][i])
            y.append(out_3d[j+1][1][i])
            z.append(out_3d[j+1][2][i])
        if out_3d[0][i][-1] == 'F':
            x[1:] = wma(x[1:])
            y[1:] = wma(y[1:])
            z[1:] = wma(z[1:])
        w.writerow(x)
        w.writerow(y)
        w.writerow(z)

# Close MediaPipe Pose
pose.close()
cap.release()