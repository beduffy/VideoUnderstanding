
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
from timeit import default_timer as timer

def process_video(video_path):
    directory = os.path.dirname(video_path)
    image_directory_path = os.path.join(directory, 'images', 'full')
    json_struct_path = os.path.join(directory, 'metadata', 'result_struct.json')

    json_struct = {'images': []}
    if os.path.isfile(json_struct_path):
        with open(json_struct_path) as data_file:
            json_struct = json.load(data_file)

    start = timer()
    filmstrip.main_separate_scenes(json_struct, video_path, False)
    end = timer()
    print 'Time taken:', round((end - start), 5), 'seconds.'

    # scene_classification.main_scene_classification(json_struct, video_path)
    # yolo_object_detection.main_object_detect(json_struct, video_path)



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

##TODO MOVE ABOVE FUNCTION TO UTILTIIES?
# TODO CREATE SAVE JSON TO FILE AND LOAD JSON FROM FILE FUNCTIONS TO UTITLITES?

process_video('/home/ben/VideoUnderstanding/example_images/Animals6mins/Animals6mins.mp4')
# create_tasks_file_from_json('/home/ben/VideoUnderstanding/example_images/Animals6mins/metadata/result_struct.json')
