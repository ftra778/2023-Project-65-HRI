import numpy as np
import matplotlib.pyplot as plt

def plot_robot_arm(joint1_coords, joint2_coords, limb_size):
    joint_coordinates = [joint1_coords, joint2_coords]
    num_joints = len(joint_coordinates)
    link_lengths = [limb_size, limb_size]  # Assuming equal link lengths

    # Calculate the x and y coordinates of each joint
    x_coords = np.cumsum(link_lengths * np.cos(joint_coordinates))
    y_coords = np.cumsum(link_lengths * np.sin(joint_coordinates))

    # Plot the robot arm
    fig, ax = plt.subplots()
    ax.plot(x_coords, y_coords, 'bo-')

    # Plot the joints
    ax.plot(x_coords, y_coords, 'ro')

    # Add labels for joints
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        ax.text(x, y, f'Joint {i+1}', ha='right', va='bottom')

    # Add labels for links
    for i in range(num_joints - 1):
        x_mid = (x_coords[i] + x_coords[i+1]) / 2
        y_mid = (y_coords[i] + y_coords[i+1]) / 2
        link_length = link_lengths[i]
        link_angle = np.arctan2(y_coords[i+1] - y_coords[i], x_coords[i+1] - x_coords[i])
        ax.text(x_mid, y_mid, f'Link {i+1}\n({link_length})', ha='center', va='bottom', rotation=link_angle * 180 / np.pi)

    # Set plot limits and aspect ratio
    ax.set_xlim(min(x_coords) - 1, max(x_coords) + 1)
    ax.set_ylim(min(y_coords) - 1, max(y_coords) + 1)
    ax.set_aspect('equal')

    # Display the plot
    plt.show()

# Example usage
joint1_coords = 0.0
joint2_coords = np.pi/4
limb_size = 2.0
plot_robot_arm(joint1_coords, joint2_coords, limb_size)
