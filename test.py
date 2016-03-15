import cv2
import cv
import argparse
import json
import os
import numpy as np
import errno

def video_step_through(video_path):
    cap = cv2.VideoCapture(video_path)

    # ret, frame = cap.read()
    # if frame == None:
    #     return

    cv2.namedWindow("Frame")
    frame_number = 0
    changeImage(cap, frame_number)

    k = 'a'
    while(k != 'q'):
        k = cv2.waitKey(1)
        if  k == ord('p'):
            frame_number += 1
            changeImage(cap, frame_number)
        if k == ord('o'):
            frame_number -= 1
            changeImage(cap, frame_number)
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def changeImage(cap, frame_number):
    cap.set(cv.CV_CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if frame == None:
        return

    text = "frame_no: " + str(frame_number)
    cv2.putText(frame, text, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
    cv2.imshow('Frame', frame)


video_step_through('/home/ben/VideoUnderstanding/example_images/Animals6mins/Animals6mins.mp4')