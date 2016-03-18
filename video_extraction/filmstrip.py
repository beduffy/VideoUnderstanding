# Scripts to try and detect key frames that represent scene transitions
# in a video. Has only been tried out on video of slides, so is likely not
# robust for other types of video.

import cv2
import cv
import argparse
import json
import os
import numpy as np
from scipy.spatial import distance
import errno
import colorsys
from skimage import io, color
from python_features import histogram

def getInfo(sourcePath):
    cap = cv2.VideoCapture(sourcePath)
    info = {
        "framecount": cap.get(cv.CV_CAP_PROP_FRAME_COUNT),
        "fps": cap.get(cv.CV_CAP_PROP_FPS),
        "width": int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv.CV_CAP_PROP_FOURCC))
    }

    cap.release()
    return info


def scale(img, xScale, yScale):
    res = cv2.resize(img, None,fx=xScale, fy=yScale, interpolation = cv2.INTER_AREA)
    return res

def resize(img, width, height):
    res = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    return res

#
# Extract [numCols] domninant colors from an image
# Uses KMeans on the pixels and then returns the centriods
# of the colors
#
def extract_cols(image, numCols):
    # convert to np.float32 matrix that can be clustered
    Z = image.reshape((-1,3))
    Z = np.float32(Z)

    # Set parameters for the clustering
    max_iter = 20
    epsilon = 1.0
    K = numCols
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)

    # cluster
    compactness, labels, centers = cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    clusterCounts = []
    for idx in range(K):
        count = len(Z[labels == idx])
        clusterCounts.append(count)

    #Reverse the cols stored in centers because cols are stored in BGR
    #in opencv.
    rgbCenters = []
    for center in centers:
        bgr = center.tolist()
        bgr.reverse()
        rgbCenters.append(bgr)

    cols = []
    for i in range(K):
        iCol = {
            "count": clusterCounts[i],
            "col": rgbCenters[i]
        }
        cols.append(iCol)

    return cols


#
# Calculates change data one one frame to the next one.
#
def calculateFrameStats(sourcePath, verbose=False, after_frame=0):
    cap = cv2.VideoCapture(sourcePath)

    data = {
        "frame_info": []
    }

    lastFrame = None
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame == None:
            break

        frame_number = cap.get(cv.CV_CAP_PROP_POS_FRAMES) - 1

        # Convert to grayscale, scale down and blur to make
        # calculate image differences more robust to noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = scale(gray, 0.25, 0.25)
        gray = cv2.GaussianBlur(gray, (9,9), 0.0)

        if frame_number < after_frame:
            lastFrame = gray
            continue


        if lastFrame != None:

            diff = cv2.subtract(gray, lastFrame)

            diffMag = cv2.countNonZero(diff)

            frame_info = {
                "frame_number": int(frame_number),
                "diff_count": int(diffMag)
            }
            data["frame_info"].append(frame_info)

            if verbose:
                text = "frame_no: " + str(frame_number) + " diff: " + str(diffMag)
                cv2.putText(diff, text, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
                cv2.imshow('diff', diff)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Keep a ref to his frame for differencing on the next iteration
        lastFrame = gray

    cap.release()
    cv2.destroyAllWindows()

    #compute some states
    diff_counts = [fi["diff_count"] for fi in data["frame_info"]]
    data["stats"] = {
        "num": len(diff_counts),
        "min": np.min(diff_counts),
        "max": np.max(diff_counts),
        "mean": np.mean(diff_counts),
        "median": np.median(diff_counts),
        "sd": np.std(diff_counts)
    }
    greater_than_mean = [fi for fi in data["frame_info"] if fi["diff_count"] > data["stats"]["mean"]]
    greater_than_median = [fi for fi in data["frame_info"] if fi["diff_count"] > data["stats"]["median"]]
    greater_than_one_sd = [fi for fi in data["frame_info"] if fi["diff_count"] > data["stats"]["sd"] + data["stats"]["mean"]]
    greater_than_two_sd = [fi for fi in data["frame_info"] if fi["diff_count"] > (data["stats"]["sd"] * 2) + data["stats"]["mean"]]
    greater_than_three_sd = [fi for fi in data["frame_info"] if fi["diff_count"] > (data["stats"]["sd"] * 3) + data["stats"]["mean"]]

    data["stats"]["greater_than_mean"] = len(greater_than_mean)
    data["stats"]["greater_than_median"] = len(greater_than_median)
    data["stats"]["greater_than_one_sd"] = len(greater_than_one_sd)
    data["stats"]["greater_than_three_sd"] = len(greater_than_three_sd)
    data["stats"]["greater_than_two_sd"] = len(greater_than_two_sd)

    return data


# Take an image and write it out at various sizes.
#
# TODO: BEN KEEP AROUND TO EVENTUALLY LOOK AT ALL SMALL IMAGES OF VIDEO FROM AFAR AND SEE IF IT IS EASY TO SPOT SCENE CHANGES FOR A HUMAN
def writeImagePyramid(destPath, name, seqNumber, image):
    fullPath = os.path.join(destPath, "full", name + "-" + str(seqNumber) + ".png")
    halfPath = os.path.join(destPath, "half", name + "-" + str(seqNumber) + ".png")
    quarterPath = os.path.join(destPath, "quarter", name + "-" + str(seqNumber) + ".png")
    eigthPath = os.path.join(destPath, "eigth", name + "-" + str(seqNumber) + ".png")
    sixteenthPath = os.path.join(destPath, "sixteenth", name + "-" + str(seqNumber) + ".png")

    hImage = scale(image, 0.5, 0.5)
    qImage = scale(image, 0.25, 0.25)
    eImage = scale(image, 0.125, 0.125)
    sImage = scale(image, 0.0625, 0.0625)

    cv2.imwrite(fullPath, image)
    cv2.imwrite(halfPath, hImage)
    cv2.imwrite(quarterPath, qImage)
    cv2.imwrite(eigthPath, eImage)
    cv2.imwrite(sixteenthPath, sImage)

#
# Selects a set of frames as key frames (frames that represent a significant difference in
# the video i.e. potential scene chnges). Key frames are selected as those frames where the
# number of pixels that changed from the previous frame are more than 1.85 standard deviations
# times from the mean number of changed pixels across all interframe changes.
#

def detect_scenes(cap, json_struct, data, verbose):
    multiplier = 1.44

    multplier_times_sd = (json_struct["stats"]["sd"] * multiplier)
    mean_plus_multiplier_times_sd = multplier_times_sd + json_struct["stats"]["mean"]

    all_chi_diffs = []

    count = 0
    for idx, fi in enumerate(data['frame_info']):
        if fi['chi_diff'] > mean_plus_multiplier_times_sd:
            right_frame_no = fi['frame_number']
            left_frame_no = data['frame_info'][idx - 1]['frame_number']

            if verbose:
                # TODO MAYBE CHANGE FRAMES TEXTS ON IMSHOW TO SAME SO LESS CLUTTER
                cap.set(cv.CV_CAP_PROP_POS_FRAMES, right_frame_no)
                ret, right_frame = cap.read()

                frame_text = ('frame_no: {0} -- chi_diff: {1}').format(right_frame_no, data['frame_info'][idx - 1]["chi_diff"])
                cv2.putText(right_frame, frame_text , (300, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                cv2.imshow('left frame', right_frame)

                # previous frame
                cap.set(cv.CV_CAP_PROP_POS_FRAMES, left_frame_no)
                ret, left_frame = cap.read()

                frame_text = ('frame_no: {0} -- chi_diff: {1}').format(left_frame_no, data['frame_info'][idx - 1]["chi_diff"])
                cv2.putText(left_frame, frame_text, (300, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.imshow('right frame', left_frame)

            # isolate down to 10 range
            still_count, left_frame_no, right_frame_no = isolate_from_100_to_10_range(cap, left_frame_no, right_frame_no)

            all_chi_diffs.append({'chi_diff': round(fi['chi_diff'], 4), 'frame_range_100': ('{0}-{1}').format(fi['frame_number'] - 100,  fi['frame_number']),
                                  'frame_range_10': ('{0}-{1}').format(left_frame_no,  right_frame_no),
                                  'sds_over_mean': round((fi['chi_diff'] - (json_struct["stats"]["mean"])) /  json_struct["stats"]["sd"], 4)})
            count += 1


    all_scene_changes_by_frame_no = sorted(all_chi_diffs, key=lambda k: int(k['frame_range_100'].split('-')[0]))
    all_scene_changes_by_chi_diff = sorted(all_chi_diffs, key=lambda k: k['chi_diff'])

    json_struct['scene_changes'] = all_scene_changes_by_frame_no

    if verbose:
        for i in all_scene_changes_by_frame_no:
            print i
        print '\n\n'

        for i in all_scene_changes_by_chi_diff:
            print i

    # TODO keep expanding this whole section so that when I change to other videos (videos with only 1-2 scenes) I can keep debugging

    print 'multiplier * sd + mean: ', str(mean_plus_multiplier_times_sd)
    print "Number of images taken: {0}. Number of images over {2} times standard deviation plus mean: {1}".format(idx, count, multiplier)

    return all_scene_changes_by_frame_no

def compute_chi_diff_on_all_interval(sourcePath, json_struct, verbose, interval):
    # stores all chi differences between every 100 images.
    data = {
        "frame_info": []
    }

    cap = cv2.VideoCapture(sourcePath)
    ret, last_frame = cap.read()

    frame_number = 0
    frame_number += interval

    while (cap.isOpened()):
        cap.set(cv.CV_CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if frame != None:
            chi_diff = histogram.chi2_distance_two_images(frame, last_frame)

            frame_info = {
                "frame_number": int(frame_number),
                "chi_diff": chi_diff
            }
            data["frame_info"].append(frame_info)

            # todo no need to keep verbose?
            # if verbose:
            #     cv2.putText(frame, "chidiff = " + str(chi_diff), (600, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
            #     cv2.imshow('frame', frame)

            k = cv2.waitKey(1)
            if k == ord('q'):
                    break

            last_frame = frame
        else:
            break

        frame_number += interval

    # Compute some stats
    chi_diff_counts = [fi["chi_diff"] for fi in data["frame_info"]]
    json_struct['stats'] = {
        "num": len(chi_diff_counts),
        "min": np.min(chi_diff_counts),
        "max": np.max(chi_diff_counts),
        "mean": np.mean(chi_diff_counts),
        "median": np.median(chi_diff_counts),
        "sd": np.std(chi_diff_counts)
    }
    greater_than_mean = [fi for fi in data["frame_info"] if fi["chi_diff"] > json_struct["stats"]["mean"]]
    greater_than_median = [fi for fi in data["frame_info"] if fi["chi_diff"] > json_struct["stats"]["median"]]
    greater_than_one_sd = [fi for fi in data["frame_info"] if fi["chi_diff"] > json_struct["stats"]["sd"] + json_struct["stats"]["mean"]]
    greater_than_two_sd = [fi for fi in data["frame_info"] if fi["chi_diff"] > (json_struct["stats"]["sd"] * 2) + json_struct["stats"]["mean"]]
    greater_than_three_sd = [fi for fi in data["frame_info"] if fi["chi_diff"] > (json_struct["stats"]["sd"] * 3) + json_struct["stats"]["mean"]]

    json_struct["stats"]["greater_than_mean"] = len(greater_than_mean)
    json_struct["stats"]["greater_than_median"] = len(greater_than_median)
    json_struct["stats"]["greater_than_one_sd"] = len(greater_than_one_sd)
    json_struct["stats"]["greater_than_three_sd"] = len(greater_than_three_sd)
    json_struct["stats"]["greater_than_two_sd"] = len(greater_than_two_sd)

    json_struct["stats"]['mean_plus_one_sd'] = (json_struct["stats"]["sd"]) + json_struct["stats"]["mean"]
    json_struct["stats"]['mean_plus_two_sd'] = (json_struct["stats"]["sd"] * 2) + json_struct["stats"]["mean"]

    # if verbose:
    #     print json.dumps(json_struct, indent=4)

    return cap, data

def save_all_relevant_frames(cap, sourcePath, destPath, name, json_struct, verbose):
    dest_dir = os.path.join(destPath, "images")

    # TODO make sd multiplier externally configurable
    # diff_threshold = (json_struct["stats"]["sd"] * 1.85) + json_struct["stats"]["mean"]
    # diff_threshold = (json_struct["stats"]["sd"] * 2.5) + json_struct["stats"]["mean"]
    diff_threshold = (json_struct["stats"]["sd"] * 3.5) + json_struct["stats"]["mean"]

    json_struct['images'] = []

    scene_num = 0

    hist_features = {}

    INITIAL_NUM_FRAMES_IN_SCENE = 5

    # todo below into function ASAP
    # FUNCTION CALLED create frames
    first_scene_first_frame = 0
    for scene_change in json_struct['scene_changes']:
        num_range = scene_change['frame_range_100'].split('-')
        next_scene_first_frame = int(num_range[0])

        num_frames_in_scene = INITIAL_NUM_FRAMES_IN_SCENE
        range = next_scene_first_frame - first_scene_first_frame
        jump_rate = range / num_frames_in_scene

        # should NEVER HAPPEN
        if jump_rate == 0:
            num_frames_in_scene = range
            jump_rate = 1

        current_frame_num = first_scene_first_frame

        print 'Saving scene frames between: ', first_scene_first_frame, '-', next_scene_first_frame
        print 'Range: ', range, 'jump rate: ', jump_rate
        print 'Scene number: ', scene_num

        frames_taken = 0
        while frames_taken < num_frames_in_scene:
            # last frame don't add???
            if current_frame_num != next_scene_first_frame:
                cap.set(cv.CV_CAP_PROP_POS_FRAMES, current_frame_num)
                ret, frame = cap.read()

                # todo separate everything into functions

                # extract dominant color
                small = resize(frame, 100, 100)
                # Todo make 5 a global?
                dom_colours = extract_cols(small, 3)

                if frame != None:
                    #todo png always?
                    image_name = name + "-" + str(current_frame_num) + ".png"

                    fullPath = os.path.join(dest_dir, "full", image_name)
                    cv2.imwrite(fullPath, frame)

                    print fullPath

                    avg_colour = [0.0, 0.0, 0.0]
                    total = 10000.0
                    for colour in dom_colours:
                        weight = colour['count'] / total
                        for idx, num in enumerate(colour['col']):
                            avg_colour[idx] += weight * num
                            # avg_colour.append(weight * num)

                    hist = histogram.calculate_hist(frame)
                    hist_features[image_name] = hist

                    json_struct['images'].append({'image_name': image_name, 'frame_number': current_frame_num, 'scene_num': scene_num,
                                                  'dominant_colours': {'kmeans' : dom_colours, 'avg_colour': {'col': avg_colour}}, })

                    # 'frame_difference_magnitude': fi["diff_count"]
                    if verbose:
                        cv2.imshow('extract', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                current_frame_num += jump_rate
                frames_taken += 1

        scene_num += 1

        first_scene_first_frame = int(num_range[1])
    json_struct['info']['num_scenes'] = scene_num - 1 # TODO double check if right?
    return hist_features

def process_video(sourcePath, destPath, name, json_struct, verbose=False, interval=100):
    cap, data = compute_chi_diff_on_all_interval(sourcePath, json_struct, verbose, interval)
    detect_scenes(cap, json_struct, data, verbose)
    hist_features = save_all_relevant_frames(cap, sourcePath, destPath, name, json_struct, verbose)
    compute_avg_col_dist_and_chi_diff(hist_features, json_struct)
    # TODO NEW FUNCTION

    json_struct['info']['num_images'] = len(json_struct['images'])
    json_struct['info']['length'] = round(json_struct['info']['framecount'] / json_struct['info']['fps'], 3)

    cap.release()
    cv2.destroyAllWindows()
    # return data # TODO ???

def compute_avg_col_dist_and_chi_diff(hist_features, json_struct):
    # compute euclidian distance from avg colours. Then computer histogram chi distance between every consecutive frame.
    num_images = len(json_struct['images'])
    for idx, image in enumerate(json_struct['images']):
        # if idx - 1 >= 0:
        #     prev_image = json_struct['images'][idx - 1]
        if idx + 1 == num_images:
            break

        next_avg_colour = json_struct['images'][idx + 1]['dominant_colours']['avg_colour']['col']
        cur_avg_colour = image['dominant_colours']['avg_colour']['col']

        # next_avg_colour = colorsys.rgb_to_hsv(next_avg_colour[0], next_avg_colour[1], next_avg_colour[2])
        # cur_avg_colour = colorsys.rgb_to_hsv(cur_avg_colour[0], cur_avg_colour[1], cur_avg_colour[2])

        next_avg_colour = rgb2lab(next_avg_colour)
        cur_avg_colour = rgb2lab(cur_avg_colour)
        dist = distance.euclidean(next_avg_colour, cur_avg_colour)
        image['dominant_colours']['l2distnext'] = round(dist, 3)

        # HISTOGRAM BELOW
        cur_hist = hist_features[image['image_name']]
        next_hist = hist_features[json_struct['images'][idx + 1]['image_name']]
        chi_dist_next = histogram.chi2_distance(cur_hist, next_hist)
        image['dominant_colours']['chi_dist_next'] = round(chi_dist_next, 3)

def rgb2lab(rgb):
    def func(t):
        if (t > 0.008856):
            return np.power(t, 1/3.0)
        else:
            return 7.787 * t + 16 / 116.0
    #Conversion Matrix
    matrix = [[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]]

    # RGB values lie between 0 to 1.0
    # rgb = [1.0, 0, 0] # RGB

    for i in rgb:
        i = i / 255.0

    cie = np.dot(matrix, rgb)

    cie[0] = cie[0] /0.950456
    cie[2] = cie[2] /1.088754

    # Calculate the L
    L = 116 * np.power(cie[1], 1/3.0) - 16.0 if cie[1] > 0.008856 else 903.3 * cie[1]

    # Calculate the a
    a = 500*(func(cie[0]) - func(cie[1]))

    # Calculate the b
    b = 200*(func(cie[1]) - func(cie[2]))

    #  Values lie between -128 < b <= 127, -128 < a <= 127, 0 <= L <= 100
    Lab = [b , a, L]

    # OpenCV Format
    L = L * 255 / 100
    a = a + 128
    b = b + 128
    return [b , a, L]

def isolate_from_100_to_10_range(cap, left_frame_no, right_frame_no):
    # cv2.putText(frame, "frame_no: " + str(frame_number) + " chidiff: " + str(fi["chi_diff"]) , (300, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    # cv2.imshow(str(fi['frame_number']), frame)
    # test_count = 0

    # Make False if error found!
    still_count = True

    # isolate down to 10 range
    range_between_frames = (right_frame_no - left_frame_no)
    middle_frame_number = left_frame_no + (range_between_frames / 2)

    cap.set(cv.CV_CAP_PROP_POS_FRAMES, left_frame_no)
    ret, left_frame = cap.read()
    cap.set(cv.CV_CAP_PROP_POS_FRAMES, right_frame_no)
    ret, right_frame = cap.read()

    while range_between_frames > 10:
        cap.set(cv.CV_CAP_PROP_POS_FRAMES, middle_frame_number)
        ret, middle_frame = cap.read()

        # todo keep comparing back to A and B and make statistics to jump out early once we are certain or we start to be uncertain

        # similarity difference distance.
        # Points A, B, C. If distance AB is smaller than BC then
        # midpoint = x. If distance between Ax is smaller than xB, repeat
        # slow because im making histogram for both of them again but ill fix in future TODO
        chi_dist_from_left_to_middle = histogram.chi2_distance_two_images(left_frame, middle_frame)
        chi_dist_from_right_to_middle = histogram.chi2_distance_two_images(right_frame, middle_frame)

        # if the left is more similar to the middle than right is to the middle then must find midpoint of BC
        if chi_dist_from_left_to_middle > chi_dist_from_right_to_middle:
            # if test_count == 0:
            #     print 'left is more similar to middle'
            #     print 'Now left frame no:', left_frame_no, ' right_frame_no: ', right_frame_no, ' middle_frame_no: ', middle_frame_number

            left_frame_no = middle_frame_number
            left_frame = middle_frame.copy()
            range_between_frames = (right_frame_no - left_frame_no)
            middle_frame_number = left_frame_no + (range_between_frames / 2)

            # if test_count == 0:
            #     print 'Now left frame no:', left_frame_no, ' right_frame_no: ', right_frame_no, ' middle_frame_no: ', middle_frame_number
        else:
            # if test_count == 0:
            #     print 'right is more similar to middle'
            #     print 'Now left frame no:', left_frame_no, ' right_frame_no: ', right_frame_no, ' middle_frame_no: ', middle_frame_number

            right_frame_no = middle_frame_number
            right_frame = middle_frame.copy()
            range_between_frames = (right_frame_no - left_frame_no)
            middle_frame_number = right_frame_no - (range_between_frames / 2)
    #         if test_count == 0:
    #             print 'Now left frame no:', left_frame_no, ' right_frame_no: ', right_frame_no, ' middle_frame_no: ', middle_frame_number
    #
    # # print
    # test_count+=1

    return (still_count, left_frame_no, right_frame_no)

def makeOutputDirs(path):
    #todo this doesn't quite work like mkdirp. it will fail

    if not os.path.isdir(os.path.join(path, "metadata")):
        os.makedirs(os.path.join(path, "metadata"))

    if not os.path.isdir(os.path.join(path, "images", "full")):
        os.makedirs(os.path.join(path, "images", "full"))


def main_separate_scenes(json_struct, video_path, verbose=True):
    # TODO rename above to process video and put process video inside here.
    directory = os.path.dirname(video_path)
    name = video_path.split('/')[-1][:-4]

    print
    print 'Video Path:', video_path
    print 'Directory Path:', directory
    print 'Name of video:', name
    print

    json_struct['info'] = getInfo(video_path)
    json_struct['info']['name'] = name

    makeOutputDirs(directory)

    # Run the extraction
    # calculateFrameStats could still be useful
    # data = calculateFrameStats(video_path, verbose, 0) # TODO AFTER FRAME USED TO BE HERE INSTEAD OF 0. WORK OUT WHAT IT IS.
    # data = detectScenes(video_path, directory, data, name, json_struct, verbose)

    process_video(video_path, directory, name, json_struct, verbose)

    json_path = os.path.join(directory, 'metadata', 'result_struct.json')
    json.dump(json_struct, open(json_path, 'w'), indent=4)

    print "Video Info: ", json_struct['info']
    print "Video scene and frame extraction complete."