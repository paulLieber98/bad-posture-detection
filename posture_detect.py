#https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import pygame #for ding sound effect

#init pygame sound mixer thing
pygame.mixer.init()

cam = cv.VideoCapture(1)

last_ding_time = 0 #later on for ding sound cooldown

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
                # print(f'ear landmark 7. x: {pose_landmarks[7].x}, y: {pose_landmarks[7].y}, z: {pose_landmarks[7].z}')
                global ear_x, ear_y, ear_z
                ear_x = pose_landmarks[7].x
                ear_y = pose_landmarks[7].y
                ear_z = pose_landmarks[7].z
            
            if len(pose_landmarks) > 11:
                # print(f'shoulder landmark 11. x: {pose_landmarks[11].x}, y: {pose_landmarks[11].y}, z: {pose_landmarks[11].z}')
                global shoulder_x, shoulder_y, shoulder_z
                shoulder_x = pose_landmarks[11].x
                shoulder_y = pose_landmarks[11].y
                shoulder_z = pose_landmarks[11].z

            if len(pose_landmarks) > 23:
                # print(f'hip landmark 23. x: {pose_landmarks[23].x}, y: {pose_landmarks[23].y}, z: {pose_landmarks[23].z}')
                global hip_x, hip_y, hip_z
                hip_x = pose_landmarks[23].x
                hip_y = pose_landmarks[23].y
                hip_z = pose_landmarks[23].z

            #creating vectors to calculate angle between ear, shoulder, and hip to see if slouching.
            #calculating vectors: shoulder to ear, shoulder to hip
            #calculating angle between vectors
            #if angle is too acute, ding
            #if angle is not acute, continue

            vector_shoulder_ear = [ear_x - shoulder_x, ear_y - shoulder_y, ear_z - shoulder_z] #using list instead of tuple bc its going to be changing constantly
            vector_shoulder_hip = [hip_x - shoulder_x, hip_y - shoulder_y, hip_z - shoulder_z]

            #calculating angle between vectors
            #mathematical formula for angle between vectors:
                # u⋅v=∥u∥⋅∥v∥⋅cosθ
            theta_angle = np.arccos((np.dot(vector_shoulder_ear, vector_shoulder_hip)) / np.linalg.norm(vector_shoulder_ear) * np.linalg.norm(vector_shoulder_hip))
            
            #now convert angle from radians to degrees
            theta_angle_degrees = np.degrees(theta_angle)
            # print(f'back posture slouch angle: {theta_angle_degrees}')

            #check cooldown before playing sound
            global last_ding_time
            current_time = time.time()
            ding_cooldown = 5 #seconds

            posture_threshold = 125 #degrees

            if theta_angle_degrees > posture_threshold and (current_time - last_ding_time) > ding_cooldown: 
                last_ding_time = current_time
                pygame.mixer.music.load('/Users/paullieber/bad-posture-detect/ding-36029.mp3')
                pygame.mixer.music.play()
                # time.sleep(5)
                print(f"bad posture !! Angle: {theta_angle_degrees:.1f} degrees")


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
    