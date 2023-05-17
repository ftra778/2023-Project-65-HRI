from roboticstoolbox import SerialLink

robot = SerialLink("../Datasheets/JULIETTEY20V171.urdf")

dh_parameters = robot.mdh()

# Print the DH parameters
for i, dh in enumerate(dh_parameters):
    print(f"DH parameters for Joint {i+1}:")
    print(f"a: {dh.a}")
    print(f"alpha: {dh.alpha}")
    print(f"d: {dh.d}")
    print(f"theta: {dh.theta}")
    print()

