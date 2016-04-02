
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
from video_extraction import scene_results
import json
import os
from timeit import default_timer as timer
import webbrowser
# from python_features.faster_rcnn_VOC_object_detection import faster_rcnn_VOC_object_detection as fast_rcnn_20
from pytube import api, exceptions
from pprint import pprint
from utilities.globals import log
import datetime, time

def process_video(video_path, video_url):
    #todo pass all below in one dictionary to save calculating every time?
    video_name = video_path.split('/')[-1][:-4]
    directory = os.path.dirname(video_path)
    image_directory_path = os.path.join(directory, 'images', 'full')
    json_struct_path = os.path.join(directory, 'metadata', 'result_struct.json')

    json_struct = {'images': []}
    if os.path.isfile(json_struct_path):
        with open(json_struct_path) as data_file:
            json_struct = json.load(data_file)

    # main processing sub functions ------------------

    filmstrip.main_separate_scenes(json_struct, video_path, False)

    # fast_rcnn_20.main_object_detect(json_struct, video_path)
    # print 'DIRECTORY after execution of fast rcnn is: ', os.getcwd()

    # scene_classification.main_scene_classification(json_struct, video_path)
    # yolo_object_detection.main_object_detect(json_struct, video_path)
    scene_results.average_all_scene_results(json_struct, json_struct_path)

    # ---------------------------------

    # Open URL in a new tab, if a browser window is already open.
    log('Opening browser tab with results')
    url = 'http://localhost:5000/video_results.html?video='
    # webbrowser.open_new_tab('file:///' + json_struct_path) #todo fix another time?
    webbrowser.open_new_tab(url + json_struct['info']['name'])

    # Save video in processed videos json.
    all_videos_json = {'videos': []}
    all_videos_json_path = '/home/ben/VideoUnderstanding/example_images/all_videos.json'
    if os.path.isfile(all_videos_json_path):
        with open(all_videos_json_path) as data_file:
            all_videos_json = json.load(data_file)

    for idx, video in enumerate(all_videos_json['videos']):
        if video['video_name'] == video_name:
            all_videos_json['videos'][idx]['last_processed_datetime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            break
    else: # if no breaks
        all_videos_json['videos'].append({'video_name': video_name,
                                          'video_url': video_url,
                                          'last_processed_datetime': datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})

    json.dump(all_videos_json, open(all_videos_json_path, 'w'), indent=4)

def download_video(url,changed_name=None):
    yt = api.YouTube(url)
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

    # set the filename:
    if changed_name:
        yt.set_filename(changed_name)

    yt.set_filename(yt.filename.replace(" ", "_"))

    log('Video Name: ', yt.filename)

    # Notice that the list is ordered by lowest resolution to highest. If you
    # wanted the highest resolution available for a specific file type, you
    # can simply do:
    log('Highest mp4 resolution video: ', yt.filter('mp4')[-1])

    if not yt.filter(extension='mp4'):
        log('No mp4 vidoes found!', color='red')
        return 'No Video Found'

    video = yt.filter('mp4')[-1]

    current_dir = os.path.dirname(os.path.realpath(__file__))
    video_folder_path = os.path.join(current_dir, 'example_images', yt.filename)
    if not os.path.isdir(video_folder_path):
        os.makedirs(video_folder_path)
        try:
            log('Downloading!', color='green')
            video.download(video_folder_path)
            log('Finished downloading!', color='green')
        except:
            #TODO TODO TODO TEST TEST TEST TEST BELOW CAREFUL in case recursive
            os.rmdir(video_folder_path)

    else:
        log('Folder and file already there:', video_folder_path, color='red')

    return yt.filename

# webbrowser.get('chromium')
# webbrowser.open('file:///' + '/home/ben/VideoUnderstanding/example_images/Walk_down_the_Times_Square_in_New_York/metadata/result_struct.json')
# open('/home/ben/VideoUnderstanding/example_images/Walk_down_the_Times_Square_in_New_York/metadata/result_struct.json')
# todo fix anotehr time?

# TODO CREATE SAVE JSON TO FILE AND LOAD JSON FROM FILE FUNCTIONS TO UTITLITES?
# TODO show full json file in browser
# TODO open browser first and display alll prints there. Then animate into each section
# TODO or open browser and refresh after each step
# TODO video class? to access globals or just keep json_struct?
# tODO FIND best frame in scene most representative for gif
#todo store port number?

# download_video('https://www.youtube.com/watch?v=0Y4r9YcLXDM')

# process_video('/home/ben/VideoUnderstanding/example_images/Walk_down_the_Times_Square_in_New_York/Walk_down_the_Times_Square_in_New_York.mp4', None)
# process_video('/home/ben/VideoUnderstanding/example_images/DogsBabies5mins/DogsBabies5mins.mp4')
# create_tasks_file_from_json('/home/ben/VideoUnderstanding/example_images/Animals6mins/metadata/result_struct.json')
# video_into_all_frames('/home/ben/VideoUnderstanding/example_images/Animals6mins/Animals6mins.mp4')

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

     Walk down the times square (https://www.youtube.com/watch?v=ezyrSKgcyJw)
     No scene changes

     Montage best of youtube snow falls:
     200-300
     900-1000
     1300-1400
     1400-1500  same scene as 900-1000
     1800-1900
     2100-2200
     2600-2700
     3400-3500 2 scene changes. tiny 40 frame scene
     4400-4500
     4700-4800
     5200-5300
     5500-5600
     5600-5700
     5700-5800
     6400-6500
     6500-6600
     7100-7200
     7400-7500
     7500-7600
     8600-8700
     9000-9100
     9600-9700
     9700-9800
     10300-10400
'''
