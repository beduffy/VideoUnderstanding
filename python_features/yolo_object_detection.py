import os
from subprocess import Popen, PIPE
import json
from utilities.globals import log, HEADER_SIZE, cd
# from utilities.globals import cd
from timeit import default_timer as timer
import cv2

# Pascal VOC 2012 dataset. It can detect the 20 Pascal object classes:

# person
# bird, cat, cow, dog, horse, sheep
# aeroplane, bicycle, boat, bus, car, motorbike, train
# bottle, chair, dining table, potted plant, sofa, tv/monitor

def main_object_detect(json_struct, video_path):
    log('Main function. Starting YOLO Object Detection', header=HEADER_SIZE - 1, color='darkturquoise')
    start = timer()

    directory = os.path.dirname(video_path)
    image_directory_path = os.path.join(directory, 'images', 'full')
    json_struct_path = os.path.join(directory, 'metadata', 'result_struct.json')

    with cd('/home/ben/Documents/darknet'):
        # darknet_command = "./darknet yolo test cfg/yolo-tiny.cfg yolo-tiny.weights"
        darknet_command = "./darknet yolo test cfg/yolo-small.cfg yolo-small.weights"
        split_command = darknet_command.split(' ')

        if os.path.isfile(json_struct_path):
            with open(json_struct_path) as data_file:
                json_struct = json.load(data_file)

        num_images = len(json_struct['images'])

        for idx, image in enumerate(json_struct['images']):
            pipe = Popen(split_command,  stdin=PIPE, stdout=PIPE, stderr=PIPE)
            path = os.path.join(image_directory_path, image['image_name']) + '\n'
            output, err = pipe.communicate(path)

            object_label_probs = output.split('\n')
            predict_message = object_label_probs[0]
            time_for_prediction = predict_message.split(' ')[-2]
            object_label_probs = object_label_probs[1:-1]

            for i, s in enumerate(object_label_probs):
                split = s.split(':')
                object_label_probs[i] = {'class': split[0], 'score': split[1][1:]}

            log("Processed image {}/{} object detection in {}s".format(idx, num_images, time_for_prediction))
            # print ("Processed image {}/{} object detection in {}s".format(idx, num_images, time_for_prediction))
            if object_label_probs:
                log('Objects detected: ', object_label_probs)

            # TODO open predictions.png save in section other than full images!!!!

            # if not image.get('object_lists'): #todo if testing yolo before rcnn
            #     log('inside here')
            image['object_lists'] = {}
            image['object_lists']['yolo_20'] = object_label_probs

        json.dump(json_struct, open(json_struct_path, 'w'), indent=4)

        end = timer()
        log('YOLO Object Detection complete', header=HEADER_SIZE, color='green')
        # log('Average Time taken per image: {}'.format(average_time_per_image))
        log('Time taken for YOLO object detection:', round((end - start), 5), 'seconds', color='chartreuse')

# json_struct_path = '/home/ben/VideoUnderstanding/example_images/Montage_-_The_Best_of_YouTubes_Mishaps_Involving_Ice_Snow_Cars_and_People/metadata/result_struct.json'
# with open(json_struct_path) as data_file:
#     json_struct = json.load(data_file)
# main_object_detect(json_struct, '/home/ben/VideoUnderstanding/example_images/Montage_-_The_Best_of_YouTubes_Mishaps_Involving_Ice_Snow_Cars_and_People/Montage_-_The_Best_of_YouTubes_Mishaps_Involving_Ice_Snow_Cars_and_People.mp4')




