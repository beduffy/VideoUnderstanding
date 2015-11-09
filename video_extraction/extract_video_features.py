import numpy as np
import os
import cv2
import python_features
from os import walk

# TODO Find best way to list all files
# files = []
# for (dirpath, dirnames, filenames) in walk(example_images):
#     files.extend(filenames)
#     break
# print files

# TODO argparse to run video
# parser = argparse.ArgumentParser()
# parser.add_argument('--video',
#                     help='path to caffe installation')

# print 'Frame width: ', cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH )
# print 'Frame height: ', cap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT )
# print 'Frames Per Second: ', cap.get(cv2.CV_CAP_PROP_FPS)
# print 'Number of frames: ', cap.get(cv2.CV_CAP_PROP_FRAME_COUNT)
# print 'Mat format: ', cap.get(cv2.CV_CAP_PROP_FORMAT)
# TODO add to loop. Is mp4 the problem?
# print 'Miliseconds into video: ', cap.get(cv2.CV_CAP_PROP_POS_MSEC)
# print 'next frame: :', cap.get(cv2.CV_CAP_PROP_POS_FRAMES)

def extract_video_into_pictures(video_name, picture_rate=10):
    video_dir = os.path.join(example_images, video_name[:-4])
    video_path = os.path.join(video_dir, video_name)

    print video_path
    cap = cv2.VideoCapture(video_path)

    picture_number = 0
    frame_number = 0

    file = open(video_dir + '/tasks.txt', 'w')

    cv2.namedWindow("Video")
    cv2.namedWindow("Image Captured")

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not frame.any(): break

        frame_number += 1
        if frame_number % picture_rate == 0:
            cv2.imshow('Image Captured', frame)
            cv2.imwrite(video_dir + '/' + video_name[:-4] + str(picture_number + 1) + '.jpg', frame)
            file.write(video_name[:-4] + str(picture_number + 1) + '.jpg' + '\n')
            picture_number += 1

        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    file.close()


current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
example_images = os.path.join(parent_dir, 'example_images')
print "Parent Directory: ", parent_dir
print "Images and Video Files Directiory: ", example_images

extract_video_into_pictures('DogsBabies5mins.mp4')