import json

def average_all_scene_results(json_struct, json_struct_path):
    json_struct['scenes'] = []

    #TODO so much horrible naming

    current_scene_num = 0
    all_averaged_results_for_scene = {'current_scene_scene_results': [[], []], 'current_scene_object_lists': {'yolo_20': [[]]}, 'average_colour': []}

    num_images = json_struct['info']['num_images']
    for idx, image in enumerate(json_struct['images']):
        scene_results1 = image['scene_results']['scene_results1']
        scene_results2 = image['scene_results']['scene_results2']
        object_list_YOLO = image['object_lists']['yolo_20']
        average_colour = image['dominant_colours']['avg_colour']['col']

        print 'scene_results1: ', scene_results1, '\n'
        print 'scene_results2: ', scene_results2, '\n'
        print 'object list YOLO: ', object_list_YOLO, '\n'
        print 'average colour: ', average_colour, '\n'

        all_averaged_results_for_scene['current_scene_scene_results'][0].append(scene_results1)
        all_averaged_results_for_scene['current_scene_scene_results'][1].append(scene_results2)
        all_averaged_results_for_scene['current_scene_object_lists']['yolo_20'].append(object_list_YOLO)
        # all_averaged_results_for_scene['current_scene_object_lists'][1].append(object_list_RCNN)
        all_averaged_results_for_scene['average_colour'].append(average_colour)
#
        # If next image doesn't exist or next image is another scene, average results
        if idx + 1 >= num_images or json_struct['images'][idx + 1]['scene_num'] != current_scene_num:
            print '\n-------------------------------------\n-------------------------------------\n-------------------------------------\n'
            average_scene_results(json_struct, all_averaged_results_for_scene, current_scene_num)

            # all_averaged_results_for_scene['current_scene_scene_results'][0][:] = [] ##todo this or below?
            all_averaged_results_for_scene = {'current_scene_scene_results': [[], []], 'current_scene_object_lists': {'yolo_20': [] }, 'average_colour': []}
            current_scene_num += 1
            print '\n-------------------------------------\n-------------------------------------\nSCENE HAS CHANGED TO: ', current_scene_num, '\n-------------------------------------\n'

    json.dump(json_struct, open(json_struct_path, 'w'), indent=4)

def average_scene_results(json_struct, all_averaged_results_for_scene, current_scene_num):
    if json_struct['info'].get('INITIAL_NUM_FRAMES_IN_SCENE'):
        num_images_in_scene = float(json_struct['info']['INITIAL_NUM_FRAMES_IN_SCENE'])
    else:
        num_images_in_scene = float(5)

    # SCENE RESULTS 1 ------------------
    average_scene_classes1 = {} #todo better name
    for image_results in all_averaged_results_for_scene['current_scene_scene_results'][0]:
        for prb_lbl in image_results:
            print prb_lbl
            average_scene_classes1[prb_lbl['label']] = average_scene_classes1.get(prb_lbl['label'], 0.0) + float(prb_lbl['probability'])
        print

    average_scene_classes1 = [(key, round(value / num_images_in_scene, 3)) for (key, value) in sorted(average_scene_classes1.items(), reverse=True, key=lambda a: a[1])]
    average_scene_classes1 = average_scene_classes1[:5]

    print 'average scene results1: \n'
    for i in average_scene_classes1:
        print i
    print

    # SCENE RESULTS 2 ------------------
    average_scene_classes2 = {} #todo better name
    for image_results in all_averaged_results_for_scene['current_scene_scene_results'][1]:
        for prb_lbl in image_results:
            print prb_lbl
            average_scene_classes2[prb_lbl['label']] = average_scene_classes2.get(prb_lbl['label'], 0.0) + float(prb_lbl['probability'])
        print

    average_scene_classes2 = [(key, round(value / num_images_in_scene, 3)) for (key, value) in sorted(average_scene_classes2.items(), reverse=True, key=lambda a: a[1])]
    average_scene_classes2 = average_scene_classes2[:5]

    print 'average scene results2: \n'
    for i in average_scene_classes2:
        print i
    print

    # YOLO OBJECT RESULTS ------------------
    print 'YOLO OBJECT LIST'
    class_occurrences = {}
    class_average_score = {}
    for object_list in all_averaged_results_for_scene['current_scene_object_lists']['yolo_20']:
        for cls_score in object_list:
            print cls_score
            class_occurrences[cls_score['class']] = class_occurrences.get(cls_score['class'], 0) + 1
            class_average_score[cls_score['class']] = class_occurrences.get(cls_score['score'], 0.0) + float(cls_score['score'])
        print

    for k, v in class_average_score.iteritems():
        # todo  ALWAYS DIVIDE BY NUM IMAGES IN SCENE? Or divide by number of occurences?
        class_average_score[k] = class_average_score[k] / num_images_in_scene

    class_occurrences = [(k, v) for k, v in class_occurrences.items()]
    class_average_score = [(k, v) for k, v in class_average_score.items()]

    print 'occurences: '
    print class_occurrences
    print 'class average score: '
    print class_average_score

    # class_occurrences_normalised = {}
    # for k, v in class_occurrences.iteritems():
    #     class_occurrences_normalised[k] = class_occurrences[k] / num_images_in_scene
    #
    # print 'class class_occurrences normalised per scene: '
    # print class_occurrences_normalised

    # RCNN OBJECT RESULTS ------------------            TODO

    # AVERAGE COLOUR RESULTS ------------------
    scene_average_rgb = [0.0, 0.0, 0.0]
    for rgb_list in all_averaged_results_for_scene['average_colour']:
        scene_average_rgb[0] += rgb_list[0]
        scene_average_rgb[1] += rgb_list[1]
        scene_average_rgb[2] += rgb_list[2]
        # print rgb_list
        # for rgb in rgb_list:
        #     print rgb[0]
        # print

    for idx, column in enumerate(scene_average_rgb):
        scene_average_rgb[idx] = scene_average_rgb[idx] / num_images_in_scene

    print 'average rgb: ', scene_average_rgb

    json_struct['scenes'].append({'scene_num': current_scene_num,
                                  'average_colour': scene_average_rgb,
                                  'scene_classes': {'scene_results1': average_scene_classes1, 'scene_results2': average_scene_classes2},
                                  'object_classes':
                                      {'yolo_20': {'class_occurrences': class_occurrences, 'class_average_score': class_average_score}}})

                                      # {'yolo_20': {'class_occurrences': class_occurrences, 'class_occurrences_normalised': class_occurrences_normalised, 'class_average_score': class_average_score}}})

    ##big question is to whether divide the results by num_images_in_scene (5)