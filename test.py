import cv2
import cv
import argparse
import json
import os
import numpy as np
import errno
from scipy.spatial import distance

def video_step_through(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_count = cap.get(cv.CV_CAP_PROP_FRAME_COUNT),

    cv2.namedWindow("Frame")
    frame_number = 0
    changeImage(cap, frame_number, frame_count)

    faster_rate = 100

    k = 'a'
    while(k != 'q'):
        # print 'hoh'
        k = cv2.waitKey(1)
        # print 'hah'
        if  k == ord('p'):
        # if  k == ord('2555904'):
            frame_number += 1
            changeImage(cap, frame_number, frame_count)
        if k == ord('o'):
        # if k == 2424832:
            frame_number -= 1
            changeImage(cap, frame_number, frame_count)

        if  k == ord('m'):
        # if  k == ord('2555904'):
            frame_number += faster_rate
            changeImage(cap, frame_number, frame_count)
        if k == ord('n'):
        # if k == 2424832:
            frame_number -= faster_rate
            changeImage(cap, frame_number, frame_count)

        if k == ord('i'):
            requested_frame = int(raw_input())
            if requested_frame >= 0 and requested_frame <= cap.get(cv.CV_CAP_PROP_FRAME_COUNT): # And less than or equal to last frame
                frame_number = requested_frame
                changeImage(cap, frame_number, frame_count)
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def changeImage(cap, frame_number, frame_count):
    if frame_number > frame_count:
        print 'frame_no:', frame_number, ' > ', frame_count
        return
    cap.set(cv.CV_CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if frame == None:
        return

    text = "frame_no: " + str(frame_number)
    cv2.putText(frame, text, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 255))
    cv2.imshow('Frame', frame)
    cv2.waitKey(1)

def optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    while(1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

                                            #prev, next, scale, levels, winsize, ite, poly_n, poly_sigma
        flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 1, 4, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        # flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        cv2.imshow('optflow',bgr)
        cv2.imshow('orig', frame2)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',bgr)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()


# optical_flow('/home/ben/VideoUnderstanding/example_images/Animals6mins/Animals6mins.mp4')

video_step_through('/home/ben/VideoUnderstanding/example_images/Best_of_2015_Cute_Funny_Animals/Best_of_2015_Cute_Funny_Animals.mp4')
# video_step_through('/home/ben/VideoUnderstanding/example_images/Best_WINFAIL_Compilation_2016__Funny_FAIL_and_WIN_Videos__January_2016/Best_WINFAIL_Compilation_2016__Funny_FAIL_and_WIN_Videos__January_2016.mp4')

# video_step_through('/home/ben/VideoUnderstanding/example_images/Funny_Videos_Of_Funny_Animals_NEW_2015/Funny_Videos_Of_Funny_Animals_NEW_2015.mp4')
# video_step_through('/home/ben/VideoUnderstanding/example_images/Montage_-_The_Best_of_YouTubes_Mishaps_Involving_Ice_Snow_Cars_and_People/Montage_-_The_Best_of_YouTubes_Mishaps_Involving_Ice_Snow_Cars_and_People.mp4')
# video_step_through('/home/ben/VideoUnderstanding/example_images/DogsBabies5mins/DogsBabies5mins.mp4')

