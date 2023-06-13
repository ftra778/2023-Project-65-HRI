import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('pose landmarker result: {}'.format(result))

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="../models/pose_landmarker_full.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...

    # Use OpenCV’s VideoCapture to start capturing from the webcam.
    # Create a loop to read the latest frame from the camera using VideoCapture#read()
   while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Send live image data to perform pose landmarking.
    # The results are accessible via the `result_callback` provided in
    # the `PoseLandmarkerOptions` object.
    # The pose landmarker must be created with the live stream mode.
    landmarker.detect_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
    
    
    