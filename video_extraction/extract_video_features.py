import numpy as np
import os
import cv2
from os import walk

current_dir = os.path.dirname(__file__)
# filename = os.path.join(current_dir, '/relative/path/to/file/you/want')

print current_dir
parent_dir = os.path.dirname(current_dir)
print parent_dir

example_images = os.path.join(parent_dir, 'example_images')
print example_images
os.listdir(example_images)

f = []
for (dirpath, dirnames, filenames) in walk(example_images):
    f.extend(filenames)
    break

print f
# print filename


# video_dir = '/home/ben/Downloads/Karpathy Neural Talk - Copy/example_images/'
# video_name = 'Super Funny Animals Compilation 2014 720p (Video Only).mp4'
# video_path = video_dir + video_name
#
# cap = cv2.VideoCapture(video_path)
# print 'Frame width: ', cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH )
# print 'Frame height: ', cap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT )
# print 'Frames Per Second: ', cap.get(cv2.CV_CAP_PROP_FPS)
# print 'Number of frames: ', cap.get(cv2.CV_CAP_PROP_FRAME_COUNT)
# print 'Mat format: ', cap.get(cv2.CV_CAP_PROP_FORMAT)
#
# picture_number = 0
# frame_count = 0
# frame_limit = 600
#
# file = open('tasks.txt', 'w')
#
# cv2.namedWindow("Video")
# cv2.namedWindow("Image Captured")
#
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not frame: break
#
#     print 'Miliseconds into video: ', cap.get(cv2.CV_CAP_PROP_POS_MSEC)
#     print 'next frame: :', cap.get(cv2.CV_CAP_PROP_POS_FRAMES)
#
#     frame_count += 1
#     if frame_count > frame_limit:
#         frame_count = 0
#         cv2.imshow('Image Captured', frame)
#         # cv2.imwrite(video_name + str(picture_number) + '.jpg', frame)
#         # file.write(video_name + str(picture_number) + '.jpg' + '\n')
#         picture_number += 1
#
#     # Display the resulting frame
#     cv2.imshow('Video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()