#https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

cam = cv.VideoCapture(1)

# Get the default frame width and height
frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))


model_path = 'pose_landmarker_full.task'

#https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the live stream mode:
#EACH TIME BODY IS DETECTED, THIS FUNCTION IS CALLED
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    
    try:
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            pose_landmarks = result.pose_landmarks[0]
            
            if len(pose_landmarks) > 7:
            #this only runs if landmark 7 exists unlike this:
            # This crashes if landmark 7 doesn't exist
            # print(pose_landmarks[7].x)  # IndexError if len < 8
                print(f'ear landmark 7. x: {pose_landmarks[7].x}, y: {pose_landmarks[7].y}, z: {pose_landmarks[7].z}')
            
            if len(pose_landmarks) > 11:
                print(f'shoulder landmark 12. x: {pose_landmarks[11].x}, y: {pose_landmarks[11].y}, z: {pose_landmarks[11].z}')
            
            if len(pose_landmarks) > 23:
                print(f'hip landmark 24. x: {pose_landmarks[23].x}, y: {pose_landmarks[23].y}, z: {pose_landmarks[23].z}')
    except:
        pass #ignore errors and continue


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with PoseLandmarker.create_from_options(options) as landmarker:
  while True:
    ret, frame = cam.read()

    #converting frames from opencv to mediapipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    #now timestamp + detect 
    frame_timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, frame_timestamp_ms) #actually detect


    # Display the captured frame
    cv.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv.destroyAllWindows()
    