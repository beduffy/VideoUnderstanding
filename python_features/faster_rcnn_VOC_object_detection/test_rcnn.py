from subprocess import Popen, PIPE
from time import sleep
import json
from utilities.globals import log, HEADER_SIZE, cd
from timeit import default_timer as timer
import os

def main_object_detect(json_struct, video_path):
    # log('Main function. Starting YOLO Object Detection', header=HEADER_SIZE - 1, color='darkturquoise')
    log('Main function. Starting Faster R-CNN Object Detection', header=HEADER_SIZE - 1, color='darkturquoise')
    start = timer()

    directory = os.path.dirname(video_path)
    image_directory_path = os.path.join(directory, 'images', 'full')
    json_struct_path = os.path.join(directory, 'metadata', 'result_struct.json')

    with cd('/home/ben/VideoUnderstanding/python_features/faster_rcnn_VOC_object_detection'):
        command = 'python faster_rcnn_VOC_object_detection.py ' + json_struct_path + ' ' + video_path
        split_command = command.split(' ')

        if os.path.isfile(json_struct_path):
            with open(json_struct_path) as data_file:
                json_struct = json.load(data_file)

        pipe = Popen(split_command,  stdin=PIPE, stdout=PIPE, stderr=PIPE)

        object_list_next = False
        object_cls_score_list = []

        while True:
            line = pipe.stdout.readline()
            if not line: break
            else:
                line = line[:-1]
                print line
                if line[:3] == '~~~':
                    object_list_next = False
                    # print 'OBJECT LIST: ', object_cls_score_list
                    log('Object list: ', object_cls_score_list)
                    object_cls_score_list = []
                if object_list_next:
                    if line[:7] == 'NOTHING':
                        continue
                    split_line = line.split()
                    object_cls_score_list.append({'class': split_line[0], 'score': split_line[1]})

                elif line[:9] == 'Detection':
                    log(line)
                    object_list_next = True

        json.dump(json_struct, open(json_struct_path, 'w'), indent=4)

        end = timer()
        log('Faster R-CNN Object Detection complete', header=HEADER_SIZE, color='green')
        log('Time taken for Faster R-CNN object detection:', round((end - start), 5), 'seconds.', color='chartreuse')



# main_object_detect('/home/ben/VideoUnderstanding/example_images/Funny_Videos_Of_Funny_Animals_NEW_2015/metadata/result_struct.json', '/home/ben/VideoUnderstanding/example_images/Funny_Videos_Of_Funny_Animals_NEW_2015/Funny_Videos_Of_Funny_Animals_NEW_2015.mp4')