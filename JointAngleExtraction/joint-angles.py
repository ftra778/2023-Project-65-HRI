# import numpy as np
# import math
# import matplotlib.pyplot as plt

# def calculate_joint_angles(shoulder_coords, elbow_coords, wrist_coords):
#     shoulder = np.array(shoulder_coords)
#     elbow = np.array(elbow_coords)
#     wrist = np.array(wrist_coords)

#     vec_shoulder_elbow = elbow - shoulder
#     vec_elbow_wrist = wrist - elbow

#     roll_angle = math.atan2(vec_shoulder_elbow[1], vec_shoulder_elbow[0])
#     pitch_angle = math.atan2(vec_elbow_wrist[1], vec_elbow_wrist[0]) - roll_angle

#     return roll_angle, pitch_angle

# def plot_robot_arm(shoulder_coords, elbow_coords, wrist_coords):
#     roll, pitch = calculate_joint_angles(shoulder_coords, elbow_coords, wrist_coords)

#     # Calculate the link lengths
#     shoulder_elbow_len = np.linalg.norm(np.array(elbow_coords) - np.array(shoulder_coords))
#     elbow_wrist_len = np.linalg.norm(np.array(wrist_coords) - np.array(elbow_coords))

#     # Calculate the coordinates of the joints
#     shoulder = np.array(shoulder_coords)
#     elbow = shoulder + shoulder_elbow_len * np.array([math.cos(roll), math.sin(roll)])
#     wrist = elbow + elbow_wrist_len * np.array([math.cos(roll + pitch), math.sin(roll + pitch)])

#     # Plot the arm
#     plt.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'b-o', label='Shoulder-Elbow')
#     plt.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'g-o', label='Elbow-Wrist')

#     # Plot the joints
#     plt.plot(shoulder[0], shoulder[1], 'ro', label='Shoulder')
#     plt.plot(elbow[0], elbow[1], 'ro', label='Elbow')
#     plt.plot(wrist[0], wrist[1], 'ro', label='Wrist')

#     # Add labels and legend
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.legend()

#     plt.axis('equal')  # Set equal aspect ratio
#     plt.grid(True)  # Add gridlines
#     plt.title('Pepper Robot Arm')

#     # Display joint angles
#     plt.text(0.1, 0.9, f'Shoulder Roll: {math.degrees(roll):.2f}°', transform=plt.gca().transAxes)
#     plt.text(0.1, 0.85, f'Shoulder Pitch: {math.degrees(pitch):.2f}°', transform=plt.gca().transAxes)

#     plt.show()

# # Example usage
# shoulder_coords = [0, 0]
# elbow_coords = [1, 0.8]
# wrist_coords = [2, 0.5]

# plot_robot_arm(shoulder_coords, elbow_coords, wrist_coords)

import math

# Function to calculate joint angles
def calculate_joint_angles(shoulder_pos, elbow_pos, wrist_pos):
    # Assuming shoulder_pos, elbow_pos, and wrist_pos are tuples of (x, y) coordinates
    
    # Calculate shoulder roll angle
    shoulder_roll = math.atan2(shoulder_pos[1] - elbow_pos[1], shoulder_pos[0] - elbow_pos[0])
    
    # Calculate shoulder pitch angle
    shoulder_pitch = math.atan2(elbow_pos[1] - wrist_pos[1], elbow_pos[0] - wrist_pos[0])
    
    # Calculate elbow yaw angle
    elbow_yaw = math.atan2(elbow_pos[1] - shoulder_pos[1], elbow_pos[0] - shoulder_pos[0])
    
    # Calculate elbow roll angle
    elbow_roll = math.atan2(wrist_pos[1] - elbow_pos[1], wrist_pos[0] - elbow_pos[0])
    
    return shoulder_roll, shoulder_pitch, elbow_yaw, elbow_roll

# Sample coordinates for testing
shoulder_pos = (0, 0)
elbow_pos = (10, 10)
wrist_pos = (20, 8)

# Calculate joint angles
shoulder_roll, shoulder_pitch, elbow_yaw, elbow_roll = calculate_joint_angles(shoulder_pos, elbow_pos, wrist_pos)

# Print the calculated joint angles
print("Shoulder Roll:", math.degrees(shoulder_roll))
print("Shoulder Pitch:", math.degrees(shoulder_pitch))
print("Elbow Yaw:", math.degrees(elbow_yaw))
print("Elbow Roll:", math.degrees(elbow_roll))

