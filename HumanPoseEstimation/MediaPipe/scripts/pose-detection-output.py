import mediapipe as mp
import csv
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Create a Pose object
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open a file for writing CSV data
with open('joint_coordinates.csv', mode='w') as csv_file:
    fieldnames = ['frame', 'joint', 'x', 'y']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Start the webcam feed
    cap = cv2.VideoCapture(0)
    frame_num = 0

    # Process video frames
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the BGR image to RGB and process it with the Pose model
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)

        # Write joint coordinates to CSV file
        for joint_num, joint in enumerate(results.pose_landmarks.landmark):
            writer.writerow({'frame': frame_num, 'joint': joint_num, 'x': joint.x, 'y': joint.y})

        # Draw the pose landmarks on the image
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show the image with pose landmarks
        cv2.imshow('Pose Estimation', frame)

        # Quit the program if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1

    # Release the webcam and close the CSV file
    cap.release()
    cv2.destroyAllWindows()
