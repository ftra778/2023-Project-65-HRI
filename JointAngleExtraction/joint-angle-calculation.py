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