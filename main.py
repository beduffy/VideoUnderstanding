
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
#  9. simple_sentence_generator(images, json_struct) // e.g. “dog is running”
#  10. complex_sentence_generator(images, json_struct) // e.g. “woman is holding a dog”
#
#  // Finally write json_struct to file and run my HTML page to display the results. Eventually I could
# display the results in OpenCV as well if needed. The difficult part is filling the json_struct.
# }