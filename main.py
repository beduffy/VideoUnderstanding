
# TODO IDEAL PIPELINE
# extract_video_features(video) {
#  json_struct = {} // Each function will fill the struct with more information and use its information.
#  1. extract_relevant_frames(video, json_struct) // Creates frames of video
#  2. extract_keyframes(images, json_struct) // Finds images different than its neighbours.
#  3. kmeans_image_batch(images, json_struct) //Function mentioned above
#  4. scene_identification(images, json_struct) // json_struct will point to correct images to identify.
#  5. image_classification(images, json_struct) //Probably useless outputting only 1 class but why not.
#  6. object_detection(images, json_struct) //Very useful
#  7. human_animal_detection(images, json_struct) //Further confirmation of humans/animals
#  8. action_recognition(images, json_struct) //Action recognition of detected objects
#  9. simple_sentence_generator(images, json_struct) // e.g. dog is running
#  10. complex_sentence_generator(images, json_struct) // e.g. woman is holding a dog
#
#  // Finally write json_struct to file and run my HTML page to display the results. Eventually I could
# display the results in OpenCV as well if needed. The difficult part is filling the json_struct.
# }

#TODO put images in folder inside folder e.g. AnimalsBabies5mins/images
# TODO HAVE TEST FUNCTION SO I DON;T HAVE TO WAIT FOR RESULTS. RUN ON OLD EXAMPLE IMAGES

from video_extraction import filmstrip
from python_features import scene_classification
from python_features import yolo_object_detection
import json
import os
import cv, cv2
from timeit import default_timer as timer
import webbrowser
from python_features.faster_rcnn_VOC_object_detection import faster_rcnn_VOC_object_detection as fast_rcnn_20
from pytube import api
# from pytube import
# not necessary, just for demo purposes.
from pprint import pprint

def process_video(video_path):
    directory = os.path.dirname(video_path)
    image_directory_path = os.path.join(directory, 'images', 'full')
    json_struct_path = os.path.join(directory, 'metadata', 'result_struct.json')

    json_struct = {'images': []}
    if os.path.isfile(json_struct_path):
        with open(json_struct_path) as data_file:
            json_struct = json.load(data_file)

    start = timer()
    #TODO PRINT OUT WHEN SCENE SEPARATION STARTED AND ALL THE FUNCTIONS WHEN THEY START ETC
    filmstrip.main_separate_scenes(json_struct, video_path, True)
    end = timer()
    print 'Time taken:', round((end - start), 5), 'seconds.'

    fast_rcnn_20.main_object_detect(json_struct, video_path)

    print 'DIRECTORY after execution of fast rcnn is: ', os.getcwd()
    # pytube_download_and_info("http://www.youtube.com/watch?v=Ik-RsDGPI5Y")

    # scene_classification.main_scene_classification(json_struct, video_path)
    # yolo_object_detection.main_object_detect(json_struct, video_path)
    # Average all results for scene()


    # TODO open browser first and display alll prints there. Then animate into each section

    # TODO or open browser and refresh after each step

    # TODO video class? to access globals or just keep json_struct?

    # tODO FIND best frame in scene most representative for gif

    url = 'http://localhost:8000/video_results.html?video='

    # Open URL in a new tab, if a browser window is already open.
    webbrowser.open_new_tab(url + json_struct['info']['name'])

    # AJAX calls and make the whole system a server type system to see real cool loading effects in JavaScript? Overkill?

def pytube_download_and_info(url):
    yt = api.YouTube(url)

    # Once set, you can see all the codec and quality options YouTube has made
    # available for the perticular video by printing videos.

    pprint(yt.get_videos())

    # [<Video: MPEG-4 Visual (.3gp) - 144p>,
    #  <Video: MPEG-4 Visual (.3gp) - 240p>,
    #  <Video: Sorenson H.263 (.flv) - 240p>,
    #  <Video: H.264 (.flv) - 360p>,
    #  <Video: H.264 (.flv) - 480p>,
    #  <Video: H.264 (.mp4) - 360p>,
    #  <Video: H.264 (.mp4) - 720p>,
    #  <Video: VP8 (.webm) - 360p>,
    #  <Video: VP8 (.webm) - 480p>]

    # The filename is automatically generated based on the video title.  You
    # can override this by manually setting the filename.

    # view the auto generated filename:
    print(yt.filename)

    # Pulp Fiction - Dancing Scene [HD]

    # set the filename:
    yt.set_filename('Dancing Scene from Pulp Fiction')

    # You can also filter the criteria by filetype.
    pprint(yt.filter('flv'))

    # [<Video: Sorenson H.263 (.flv) - 240p>,
    #  <Video: H.264 (.flv) - 360p>,
    #  <Video: H.264 (.flv) - 480p>]

    # Notice that the list is ordered by lowest resolution to highest. If you
    # wanted the highest resolution available for a specific file type, you
    # can simply do:
    print(yt.filter('mp4')[-1])
    # <Video: H.264 (.mp4) - 720p>

    # You can also get all videos for a given resolution
    pprint(yt.filter(resolution='480p'))

    # [<Video: H.264 (.flv) - 480p>,
    #  <Video: VP8 (.webm) - 480p>]

    # To select a video by a specific resolution and filetype you can use the get
    # method.

    # video = yt.get('mp4', '720p')

    # NOTE: get() can only be used if and only if one object matches your criteria.
    # for example:

    # pprint(yt.get_videos())

    #[<Video: MPEG-4 Visual (.3gp) - 144p>,
    # <Video: MPEG-4 Visual (.3gp) - 240p>,
    # <Video: Sorenson H.263 (.flv) - 240p>,
    # <Video: H.264 (.flv) - 360p>,
    # <Video: H.264 (.flv) - 480p>,
    # <Video: H.264 (.mp4) - 360p>,
    # <Video: H.264 (.mp4) - 720p>,
    # <Video: VP8 (.webm) - 360p>,
    # <Video: VP8 (.webm) - 480p>]

    # Since we have two H.264 (.mp4) available to us... now if we try to call get()
    # on mp4...

    video = yt.get('mp4', '720p')
    # MultipleObjectsReturned: 2 videos met criteria.

    # In this case, we'll need to specify both the codec (mp4) and resolution
    # (either 360p or 720p).

    # Okay, let's download it!
    print 'downloading!'
    # video.download()

    # If you wanted to choose the output directory, simply pass it as an
    # argument to the download method.
    video.download('example_images')

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


##TODO MOVE ABOVE FUNCTION TO UTILTIIES?
# TODO CREATE SAVE JSON TO FILE AND LOAD JSON FROM FILE FUNCTIONS TO UTITLITES?
# TODO  show full json file in browser

# process_video('/home/ben/VideoUnderstanding/example_images/Animals6mins/Animals6mins.mp4')

process_video('/home/ben/VideoUnderstanding/example_images/DogsBabies5mins/DogsBabies5mins.mp4')

# create_tasks_file_from_json('/home/ben/VideoUnderstanding/example_images/Animals6mins/metadata/result_struct.json')


# video_into_all_frames('/home/ben/VideoUnderstanding/example_images/Animals6mins/Animals6mins.mp4')


#'/home/ben/VideoUnderstanding/example_images/Animals6mins/Animals6mins.mp4'

'''
    Animals6mins
    17 scene changes
    key frames for above:
    201-211
    971-981
    1561-1571
    2261-2271
    2781-2791
    3211-3221
    3781-3791
    4581-4591
    5561-5571
    6211-6221
    6751-6761
    7231-7241
    7951-7961
    8791-8801
    9931-8941 blackness of arm
    9571-9581
    10281-10291

    DogsBabies5mins
    100-200
    2000-2100
    2600-2700
    4700-4800
    7000-7100
    8400-8500

    multiplier 1.44 missing 4
    multiplier 1.25 missing 1 scene change
    0.676 missine none.
    Loads of extra at end but not saving them???
'''