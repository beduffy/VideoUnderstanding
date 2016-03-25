# from flask import Flask
import os
from flask_socketio import SocketIO
# s = None

class SocketPrint:
    socketio = None

    def __init__(self, socket):
        socketio = socket

    def s_print(self, str):
        print 'calling s_print'
        print str
        socketio.emit('print_event', str)

def init_globals(app):
    # global app = Flask(__name__, static_url_path='')
    # app.config['SECRET_KEY'] = 'secret!'
    # app.debug = True
    global socketio
    socketio = SocketIO(app)
    global s
    s = SocketPrint(socketio)


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

def log(*args):
    print 'inside log'
    # todo might not need below 3 lines can just pass args to join?
    str_list = []

    # print 'length: ', len(args)

    for i in args:
        str_list.append(str(i))

    ret_str = ' '
    ret_str = ret_str.join(str_list)

    print ret_str
    socketio.emit('print_event', ret_str)
    # print str
    # socketio.emit('print_event', str)


