import os, json


def calculate_accuracy(video_name, keyframe_ground_truth_json):
    example_images = '/home/ben/VideoUnderstanding/example_images'
    json_struct_path = os.path.join(example_images, video_name, 'metadata', 'result_struct.json')

    json_struct = {}
    if os.path.isfile(json_struct_path):
        with open(json_struct_path) as data_file:
            json_struct = json.load(data_file)
    else:
        print 'No file found!'
        return

    # Find ground truth results
    ground_truth_results = None
    for video in keyframe_ground_truth_json['videos']:
        if video['video_name'] == video_name:
            ground_truth_results = video['results']
            break

    # # print ground truth results
    # print 'GROUND TRUTH'
    # for range in ground_truth_results:
    #     print range
    #
    # # print video keyframe results
    # print '\nRESULTS'
    # for scene_change in json_struct['scene_changes']:
    #     print scene_change['keyframe_range']

    # print '\nANSWERS --- GROUND TRUTH --- RESULT\n'

    num_ground_truth_scene_changes = len(ground_truth_results)
    num_result_scene_changes = len(json_struct['scene_changes'])

    # In GT + RS = True positive
    # In GT NOT in RS = False negative
    # Not in GT in RS = False Positive
    num_true_positives, num_false_positives, num_false_negatives = 0, 0, 0

    found_ground_truth_ranges = []
    for range in json_struct['scene_changes']:
        result_range = range['keyframe_range'].split('-')

        found = False
        for ground_range in ground_truth_results:
            split_ground_truth_range = ground_range.split('-')

            if int(result_range[0]) >= int(split_ground_truth_range[0]) and int(result_range[1]) <= int(split_ground_truth_range[1]):
                # print result_range[0], result_range[1], ' --- ', split_ground_truth_range[0], split_ground_truth_range[1], ' --- True Positive'
                found = True
                num_true_positives += 1
                found_ground_truth_ranges.append(ground_range)
                break

        if not found:
            # print result_range[0], result_range[1], ' --- False Positive'
            num_false_positives += 1

    # Ranges that are in ground truth but were not in the results

    #todo make sure if it is right

    false_negatives = list(set(ground_truth_results) - set(found_ground_truth_ranges))

    # todo horribly hacky because 2 scene changes within 100 frame range in ground truth
    if video_name == 'Montage_-_The_Best_of_YouTubes_Mishaps_Involving_Ice_Snow_Cars_and_People':
        num_false_negatives += 1

    num_false_negatives = len(false_negatives)

    print '\nEvaluating results for video: ', video_name

    print '\nRanges in ground truth but not inside results (False Negatives): ', false_negatives
    print 'Number of ground truth scenes changes: ', num_ground_truth_scene_changes
    print 'Number of result scene changes: ', num_result_scene_changes, '\n'

    print 'Number of true positives: ', num_true_positives
    print 'Number of false positives: ', num_false_positives
    print 'Number of false negatives: ', num_false_negatives

    # no true negatives
    # Accuracy = (TP + TN) / (TP + FP + TN + FN)
    accuracy = (num_true_positives) / float(num_true_positives + num_false_positives + num_false_negatives)
    # Precision = TP / (TP + FP)
    precision = (num_true_positives) / float(num_true_positives + num_false_positives)
    # Recall TP / (TP + FN)
    recall = (num_true_positives) / float(num_true_positives + num_false_negatives)

    print 'Accuracy: ', accuracy
    print 'Precision: ', precision
    print 'Recall: ', recall

# calculate_accuracy('Funny_Videos_Of_Funny_Animals_NEW_2015', keyframe)

all_videos_json_path = '/home/ben/VideoUnderstanding/example_images/all_videos.json'
if os.path.isfile(all_videos_json_path):
    with open(all_videos_json_path) as data_file:
        all_videos_json = json.load(data_file)

all_videos_names = [i['video_name'] for i in all_videos_json['videos']]
print all_videos_names

keyframe_ground_truth_path = '/home/ben/VideoUnderstanding/example_images/keyframe_ground_truth.json'
if os.path.isfile(keyframe_ground_truth_path):
    with open(keyframe_ground_truth_path) as data_file:
        keyframe_ground_truth_json = json.load(data_file)

all_result_video_names = [i['video_name'] for i in keyframe_ground_truth_json['videos'] if i.get('results')]
print all_result_video_names

for name in all_result_video_names:
    calculate_accuracy(name, keyframe_ground_truth_json)