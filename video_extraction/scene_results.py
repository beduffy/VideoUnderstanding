

def average_all_scene_results(json_struct):
    json_struct['scenes'] = []

    #TODO so much horrible naming

    current_scene_num = 0
    all_averaged_results_for_scene = {'current_scene_scene_results': [[], []], 'current_scene_object_lists': [[], []], 'average_colour': []}

    for idx, image in enumerate(json_struct['images']):
        if image['scene_num'] != current_scene_num:
            # alist[:] = []


            #
            print '\n-------------------------------------\n-------------------------------------\n-------------------------------------\n'


            average_scene_results(json_struct, all_averaged_results_for_scene)



            # all_averaged_results_for_scene['current_scene_scene_results'][0][:] = [] ##todo this or below?
            all_averaged_results_for_scene = {'current_scene_scene_results': [[], []], 'current_scene_object_lists': [[], []], 'average_colour': []}
            current_scene_num += 1
            print '\n-------------------------------------\n-------------------------------------\nSCENE HAS CHANGED TO: ', current_scene_num, '\n-------------------------------------\n'

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
        all_averaged_results_for_scene['current_scene_object_lists'][0].append(object_list_YOLO)
        # all_averaged_results_for_scene['current_scene_object_lists'][1].append(object_list_RCNN)
        # all_averaged_results_for_scene['current_scene_object_lists'][0].append(object_list_YOLO)
        all_averaged_results_for_scene['average_colour'].append(average_colour)

    pass


def average_scene_results(json_struct, all_averaged_results_for_scene):
    num_images_in_scene = 5.0

    average_results = {} #todo better name
    for image_results in all_averaged_results_for_scene['current_scene_scene_results'][0]:
        for prb_lbl in image_results:
            print prb_lbl
            average_results[prb_lbl['label']] = average_results.get(prb_lbl['label'], 0.0) + float(prb_lbl['probability'])
        print

    average_results = [(key, round(value / num_images_in_scene, 3)) for (key, value) in sorted(average_results.items(), reverse=True, key=lambda a: a[1])]
    average_results = average_results[:5]

    print 'average scene results1: \n'
    for i in average_results:
        print i
    print

    average_results = {} #todo better name
    for image_results in all_averaged_results_for_scene['current_scene_scene_results'][1]:
        for prb_lbl in image_results:
            print prb_lbl
            average_results[prb_lbl['label']] = average_results.get(prb_lbl['label'], 0.0) + float(prb_lbl['probability'])
        print

    average_results = [(key, round(value / num_images_in_scene, 3)) for (key, value) in sorted(average_results.items(), reverse=True, key=lambda a: a[1])]
    average_results = average_results[:5]

    print 'average scene results2: \n'
    for i in average_results:
        print i
    print

    ##big question is to whether divide the results by 5