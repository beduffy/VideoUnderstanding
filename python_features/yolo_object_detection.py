import os
import shlex,subprocess
from subprocess import Popen, PIPE
from time import sleep
import json

def main_object_detect(json_struct, video_path):
    directory = os.path.dirname(video_path)
    image_directory_path = os.path.join(directory, 'images', 'full')
    json_struct_path = os.path.join(directory, 'metadata', 'result_struct.json')

    os.chdir('/home/ben/darknet')
    # darknet_command = "./darknet yolo test cfg/yolo-tiny.cfg yolo-tiny.weights /home/ben/VideoUnderstanding/example_images/Animals6mins/images/full/Animals6mins-551.png"
    darknet_command = "./darknet yolo test cfg/yolo-tiny.cfg yolo-tiny.weights"
    split_command = darknet_command.split(' ')

    with open('/home/ben/VideoUnderstanding/example_images/Animals6mins/metadata/tasks.txt') as f:
        # lines = f.readlines()
        lines = [i[:-1] for i in f.readlines()]

    json_struct_path = os.path.join(directory, 'metadata', 'result_struct.json')
    if os.path.isfile(json_struct_path):
        with open(json_struct_path) as data_file:
            json_struct = json.load(data_file)

    current_time = 0.0
    num_images = len(json_struct['images'])

    print num_images
    idx = 0

    print 'before json_struct:', json_struct

    for line, image in zip(lines, json_struct['images']):
        pipe = Popen(split_command,  stdin=PIPE, stdout=PIPE, stderr=PIPE)
        path =  os.path.join(image_directory_path, line) + '\n'
        path = image_directory_path + '/' + line
        # path = '/home/ben/VideoUnderstanding/example_images/Animals6mins/images/full/Animals6mins-551.png'
        output, err = pipe.communicate(path)

        object_label_probs = output.split('\n')
        predict_message = object_label_probs[0]
        time_for_prediction = predict_message.split(' ')[-2]
        current_time += float(time_for_prediction)  #TODO ERROR HERE!!! ValueError: could not convert string to float: Path:
        object_label_probs = object_label_probs[1:-1]
        # object_lists.append(object_label_probs)

        # print predict_message
        # print "prediction time: ", time_for_prediction
        print "total time taken in s: {} scene {}/{}".format(current_time, idx, num_images)
        # print ' % (idx. num_images)
        # print object_label_probs

        image['object_lists'] = object_label_probs
        # TODO CHANGE ABOVE TO image['object_lists'] = [object_label_probs]
        # print image['object_lists']
        # print image

        # print 'sleeping'
        idx += 1
        print
        # sleep(0.04)

    print 'after json_struct:', json_struct
    json.dump(json_struct, open(json_struct_path, 'w'), indent=4)


# main_object_detect('/home/ben/VideoUnderstanding/example_images/Animals6mins/Animals6mins.mp4')




