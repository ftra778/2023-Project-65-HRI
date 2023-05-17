# This script will calculate the 2D joint angles between the shoulder and the elbow, and the elbow and the wrist, given the coordinates of those joints
# using inverse kinematics.

# Pepper robot dimensions:

# Shoulder joint to elbow joint = 28 cm 
# Elbow joint to wrist joint = 27 cm

# Hello

from collections.abc import Mapping
import urdfpy

urdf_file = "../Datasheets/JULIETTEY20V171.urdf"

robot = urdfpy.URDF.load(urdf_file)

# Print the name of the robot
print("Robot Name:", robot.name)

# Print the number of links and joints in the robot
print("Number of Links:", len(robot.links))
print("Number of Joints:", len(robot.joints))

# Print the information about each link in the robot
for link in robot.links:
    print("\nLink Name:", link.name)
    print("Link Mass:", link.mass)
    print("Link Inertia Tensor:", link.inertia)
    print("Link Visuals:", link.visuals)
    print("Link Collisions:", link.collisions)

# Print the information about each joint in the robot
for joint in robot.joints:
    print("\nJoint Name:", joint.name)
    print("Joint Type:", joint.joint_type)
    print("Joint Parent:", joint.parent)
    print("Joint Child:", joint.child)
    print("Joint Origin:", joint.origin)
    print("Joint Axis:", joint.axis)


# import pybullet as p
# import numpy as np

# # Load the URDF file
# robot_id = p.loadURDF("path/to/your/urdf/file.urdf")

# # Set the joint angles for a specific configuration
# joint_angles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # Replace with your desired joint angles
# p.setJointMotorControlArray(robot_id, range(len(joint_angles)), p.POSITION_CONTROL, targetPositions=joint_angles)

# # Forward Kinematics
# end_effector_link_index = len(joint_angles) - 1
# end_effector_state = p.getLinkState(robot_id, end_effector_link_index)
# end_effector_position = end_effector_state[0]
# end_effector_orientation = end_effector_state[1]

# # Inverse Kinematics
# target_position = [0.1, 0.2, 0.3]  # Replace with your desired target position
# target_orientation = p.getQuaternionFromEuler([0.1, 0.2, 0.3])  # Replace with your desired target orientation
# joint_angles = p.calculateInverseKinematics(robot_id, end_effector_link_index, target_position, target_orientation)

# # Compute Transformation Matrix
# link_index = 2  # Replace with the index of the desired link
# link_state = p.getLinkState(robot_id, link_index)
# link_position = link_state[0]
# link_orientation = link_state[1]
# transformation_matrix = p.getMatrixFromQuaternion(link_orientation)
# transformation_matrix = np.reshape(transformation_matrix, (3, 3))
# transformation_matrix[:3, 3] = link_position

# # Print the results
# print(f"End Effector Position: {end_effector_position}")
# print(f"End Effector Orientation: {end_effector_orientation}")
# print(f"Inverse Kinematics Joint Angles: {joint_angles}")
# print(f"Transformation Matrix for Link {link_index}:")
# print(transformation_matrix)
