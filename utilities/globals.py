# from flask import Flask
import os
from flask_socketio import SocketIO, emit
import cv, cv2
import json

HEADER_SIZE = 3

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def init_globals(app):
    global socketio
    socketio = SocketIO(app)

def create_tasks_file_from_json(json_struct_path):
    directory = os.path.dirname(json_struct_path)

    json_struct = {}
    with open(json_struct_path) as data_file:
        json_struct = json.load(data_file)

    tasks_path =  os.path.join(directory, 'tasks.txt')

    file = open(tasks_path, 'w+')
    num_images = len(json_struct['images'])
    for idx, image in enumerate(json_struct['images']):
        file.write(image['image_name']+'\n')
        print image['image_name']

    file.close()

def video_into_all_frames(video_path, interval=10):
    directory = os.path.dirname(video_path)
    name = video_path.split('/')[-1][:-4]

    if not os.path.isdir(os.path.join(directory, "all_frames")):
        os.makedirs(os.path.join(directory, "all_frames"))

    dest_dir = os.path.join(directory, "all_frames")

    cap = cv2.VideoCapture(video_path)
    frame_number = 1

    while (cap.isOpened()):
        cap.set(cv.CV_CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if frame != None:
            cv2.imshow('frame', frame)
            #TODO CHANGE ALL FI BELOW TO PROPER FRAME
            #writeImagePyramid(destDir, name, fi["frame_number"], frame)
            #todo png always?
            image_name = name + "-" + str(frame_number) + ".png"

            fullPath = os.path.join(dest_dir, image_name)
            cv2.imwrite(fullPath, frame)

            print fullPath
        else:
            break

        frame_number += interval

    cap.release()
    cv2.destroyAllWindows()

def log(*args, **kwargs):
    # todo might not need below 3 lines can just pass args to join?
    str_list = []

    for i, arg in enumerate(args):
        str_list.append(str(arg))

    ret_str = ' '.join(str_list)

    print ret_str

    data_to_send = {'s': ret_str}

    if not kwargs:
        data_to_send['color'] = 'white'
    else:
        # print 'kwargs', kwargs
        data_to_send['color'] = kwargs.get('color', 'white')
        # print 'data_to_send', data_to_send
        # if kwargs['header']:
        #     print 'data_to_send', data_to_send
        data_to_send['header'] = kwargs.get('header' , None)

    # print 'data_to_send', data_to_send
    socketio.emit('print_event', data_to_send)


