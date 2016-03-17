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
def detectScenes(sourcePath, destPath, data, name, json_struct, verbose=False):
    destDir = os.path.join(destPath, "images")

    # TODO make sd multiplier externally configurable
    # diff_threshold = (data["stats"]["sd"] * 1.85) + data["stats"]["mean"]
    # diff_threshold = (data["stats"]["sd"] * 2.5) + data["stats"]["mean"]
    diff_threshold = (data["stats"]["sd"] * 3.5) + data["stats"]["mean"]

    json_struct['images'] = [] # TODO OOOOOOO BIG MOVE

    scene_num = 0

    first_scene_first_frame = None
    next_scene_first_frame = None

    cap = cv2.VideoCapture(sourcePath)
    for index, fi in enumerate(data["frame_info"]):
        if fi["diff_count"] < diff_threshold:
            continue

        if not first_scene_first_frame:
            first_scene_first_frame = fi["frame_number"]
            continue

        else:
            next_scene_first_frame = fi["frame_number"]

            num_frames_in_scene = 5

            range = next_scene_first_frame - first_scene_first_frame

            jump_rate = range / num_frames_in_scene

            if jump_rate == 0:
                num_frames_in_scene = range
                jump_rate = 1

            current_frame_num = first_scene_first_frame

            print 'Saving scene frames between: ', first_scene_first_frame, '-', next_scene_first_frame
            print 'Range: ', range, 'jump rate: ', jump_rate
            print 'Scene number: ', scene_num

            frames_taken = 0
            while frames_taken < num_frames_in_scene:
                # todo have to really make sure no duplicates
                # todo last scene don't add
                # if frames_taken == num_frames_in_scene - 1:
                #     current_frame_num = first_scene_first_frame

                # last frame don't add
                if current_frame_num != next_scene_first_frame:
                    cap.set(cv.CV_CAP_PROP_POS_FRAMES, current_frame_num)
                    ret, frame = cap.read()

                    # todo all below into function
                    # extract dominant color
                    small = resize(frame, 100, 100)
                    # Todo make 5 a global?
                    dom_colours = extract_cols(small, 3)
                    #  todo for now have k means data in other json file??
                    # data["frame_info"][index]["dominant_cols"] = cols



                    if frame != None:
                        #TODO CHANGE ALL FI BELOW TO PROPER FRAME
                        #writeImagePyramid(destDir, name, fi["frame_number"], frame)
                        #todo png always?
                        image_name = name + "-" + str(current_frame_num) + ".png"

                        fullPath = os.path.join(destDir, "full", image_name)
                        cv2.imwrite(fullPath, frame)

                        print fullPath

                        avg_colour = [0.0, 0.0, 0.0]
                        total = 10000.0
                        for colour in dom_colours:
                            weight = colour['count'] / total
                            for idx, num in enumerate(colour['col']):
                                avg_colour[idx] += weight * num
                                # avg_colour.append(weight * num)

                        # print avg_colour

                        json_struct['images'].append({'image_name': image_name, 'frame_number': current_frame_num, 'scene_num': scene_num, 'dominant_colours': {'kmeans' : dom_colours, 'avg_colour': {'col': avg_colour}}})

                        if verbose:
                            cv2.imshow('extract', frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                    current_frame_num += jump_rate
                    frames_taken += 1

            first_scene_first_frame = next_scene_first_frame
            scene_num += 1

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

        print 'next colour', next_avg_colour

        next_avg_colour = rgb2lab(next_avg_colour)
        cur_avg_colour = rgb2lab(cur_avg_colour)

        print 'next colour lab', next_avg_colour
        # colorsys.
        # dist = np.linalg.norm(next_avg_colour - cur_avg_colour)
        dist = distance.euclidean(next_avg_colour, cur_avg_colour)

        image['dominant_colours']['l2distnext'] = round(dist, 3)


    json_struct['info']['num_images'] = len(json_struct['images'])
    json_struct['info']['length'] = round(json_struct['info']['framecount'] / json_struct['info']['fps'], 3)
    json_struct['info']['num_scenes'] = scene_num - 1 # TODO double check if right?
    cap.release()
    cv2.destroyAllWindows()
    return data

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

def makeOutputDirs(path):
    #todo this doesn't quite work like mkdirp. it will fail

    if not os.path.isdir(os.path.join(path, "metadata")):
        os.makedirs(os.path.join(path, "metadata"))

    if not os.path.isdir(os.path.join(path, "images", "full")):
        os.makedirs(os.path.join(path, "images", "full"))

    # os.makedirs(os.path.join(path, "images", "half"))
    # os.makedirs(os.path.join(path, "images", "quarter"))
    # os.makedirs(os.path.join(path, "images", "eigth"))
    # os.makedirs(os.path.join(path, "images", "sixteenth"))


# TODO understand after frame.

def main_separate_scenes(json_struct, video_path, verbose=True):
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('-s','--source', help='source file', required=True)
    # parser.add_argument('-d', '--dest', help='dest folder', required=True)
    # parser.add_argument('-n', '--name', help='image sequence name', required=True)
    # parser.add_argument('-a','--after_frame', help='after frame', default=0)
    # parser.add_argument('-v', '--verbose', action='store_true')
    # parser.set_defaults(verbose=False)
    #
    # args = parser.parse_args()

    directory = os.path.dirname(video_path)
    name = video_path.split('/')[-1][:-4]

    print 'Video Path:', video_path
    print 'Directory Path:', directory
    print 'Name of video:', name

    # if verbose:
    info = getInfo(video_path)


    # TODO STORE ANY INFO I CAN INSIDE JSON STRUCT SO I CAN SHOW IN HTML.

    json_struct['info'] = info

    makeOutputDirs(directory)

    # Run the extraction
    data = calculateFrameStats(video_path, verbose, 0) # TODO AFTER FRAME USED TO BE HERE INSTEAD OF 0. WORK OUT WHAT IT IS.
    data = detectScenes(video_path, directory, data, name, json_struct, verbose)
    # keyframeInfo = [frame_info for frame_info in data["frame_info"] if "dominant_cols" in frame_info] # todo doesnt include all keyframes coz no dominant cols

    # # Write out the results
    # data_fp = os.path.join(directory, "metadata", name + "-meta.json")
    # with open(data_fp, 'w') as f:
    #     data_json_str = json.dumps(data, indent=4)
    #     f.write(data_json_str)
    #
    # keyframe_info_fp = os.path.join(directory, "metadata", name + "-keyframe-meta.json")
    # with open(keyframe_info_fp, 'w') as f:
    #     data_json_str = json.dumps(keyframeInfo, indent=4)
    #     f.write(data_json_str)

    json_path = os.path.join(directory, 'metadata', 'result_struct.json')
    json.dump(json_struct, open(json_path, 'w'))

    print "Video Info: ", json_struct['info']
    print "Video scene and frame extraction complete."