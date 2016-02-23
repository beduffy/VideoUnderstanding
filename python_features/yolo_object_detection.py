import os
import shlex,subprocess
from subprocess import Popen, PIPE

def main_object_detect(video_path):
    directory = os.path.dirname(video_path)
    # subprocess.Popen(r'/home/ben/VideoUnderstanding/python_features/extract_features.py')
    os.chdir('/home/ben/darknet')
    darknet_command = "./darknet yolo test cfg/yolo-tiny.cfg yolo-tiny.weights /home/ben/VideoUnderstanding/example_images/Animals6mins/images/full/Animals6mins-551.png"
    darknet_command = "./darknet yolo test cfg/yolo-tiny.cfg yolo-tiny.weights"
    split_command = darknet_command.split(' ')

    pipe = Popen(split_command,  stdin=PIPE, stdout=PIPE, stderr=PIPE)
    # pipe.stdin.write("input")
    # pipe.stdout.readline()
    output, err = pipe.communicate(b"/home/ben/VideoUnderstanding/example_images/Animals6mins/images/full/Animals6mins-551.png\n")
    rc = pipe.returncode


    # print 'output:', output
    # stdout_data = pipe.communicate(input='/home/ben/VideoUnderstanding/example_images/Animals6mins/images/full/Animals6mins-551.png\n')[0]
    # print(stdout_data.decode())
    # print '/home/ben/VideoUnderstanding/example_images/Animals6mins/images/full/Animals6mins-551.png'
    # print 'output:', output
    # print 'err:', err

    object_label_probs = output.split('\n')
    predict_message = object_label_probs[0]

    time_for_prediction = predict_message.split(' ')[-2]

    print predict_message
    print "prediction time: ", time_for_prediction
    object_label_probs = object_label_probs[1:-1]
    print object_label_probs

    # pipe.stdin.open()


    retval = pipe.wait()
    print 'retval:', retval
    # pipe.stdout.readline()

    output, err = pipe.communicate(b"/home/ben/VideoUnderstanding/example_images/Animals6mins/images/full/Animals6mins-551.png")

    print 'output:', output





main_object_detect("bla")