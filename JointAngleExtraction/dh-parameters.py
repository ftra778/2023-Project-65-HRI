import urdfpy

urdf_path = '../Datasheets/JULIETTEY20V171.urdf'
robot = urdfpy.URDF.load(urdf_path)

print("DH Parameters:")
print("-------------------------")
for joint in robot.joints:
    if joint.type == 'revolute' or joint.type == 'continuous':
        dh_params = joint.origin.xyz, joint.origin.rpy, joint.axis, joint.parent, joint.child
        print("Joint: ", joint.name)
        print("Origin XYZ: ", dh_params[0])
        print("Origin RPY: ", dh_params[1])
        print("Axis: ", dh_params[2])
        print("Parent: ", dh_params[3])
        print("Child: ", dh_params[4])
        print("-------------------------")