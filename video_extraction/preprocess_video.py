# Utilities for preprocessing video for scene detection.
# Frame skipping: will create a video that only one frame
# for each section of the original video.
# ROI extraction: which will create a video that only
# contains the specified region of interest.

import math
import cv2
import cv
import argparse

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

#
# Extracts one frame for every second second of video.
# Effectively compresses a video down into much less data.
#
def extractFrames(sourcePath, destPath, verbose=False):
    info = getInfo(sourcePath)

    cap = cv2.VideoCapture(sourcePath)
    fourcc = cv2.cv.CV_FOURCC('X','V','I','D')
    # fourcc = cv2.cv.CV_FOURCC('M','P','4','3')
    fourcc = cv2.cv.FOURCC('8', 'B', 'P', 'S')
    out = cv2.VideoWriter(destPath,
        fourcc,
        info["fps"],
        (info["width"], info["height"]))

    ret = True
    while(cap.isOpened() and ret):
        ret, frame = cap.read()
        frame_number = cap.get(cv.CV_CAP_PROP_POS_FRAMES) - 1
        # print frame_number
        if frame_number % math.ceil(info["fps"]) == 0:
            out.write(frame)

            if verbose:
                cv2.imshow('frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    out.release()
    out = None
    cv2.destroyAllWindows()


#
# Extracts a region of interest defined by a rect
# from a video
#
def extractROI(sourcePath, destPath, points, verbose=False):
    info = getInfo(sourcePath)
    # x, y, width, height = cv2.boundingRect(points)

    # print(x,y,width,height)/
    x = points[0][0]
    y = points[0][1]

    width = points[1][0] - points[0][0]
    height = points[1][1] - points[0][1]

    cap = cv2.VideoCapture(sourcePath)

    fourcc = cv2.cv.CV_FOURCC('X','V','I','D')
    out = cv2.VideoWriter(destPath,
        fourcc,
        info["fps"],
        (width, height))

    ret = True
    while(cap.isOpened() and ret):
        ret, frame = cap.read()
        if frame == None:
            break

        roi = frame[y:y+height, x:x+width]

        out.write(roi)

        if verbose:
            cv2.imshow('frame', roi)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Groups a list in n sized tuples
def group(lst, n):
    return zip(*[lst[i::n] for i in range(n)])

parser = argparse.ArgumentParser(description='Extract one frame from every second of video')

parser.add_argument('--source', help='source file', required=True)
parser.add_argument('--dest', help='source file', required=True)
parser.add_argument('--command', help='command to run', required=True)
parser.add_argument('--rect', help='x1,y2,x2,y2', required=False)
parser.add_argument('--verbose', action='store_true')
parser.set_defaults(verbose=False)

args = parser.parse_args()

if args.verbose:
    info = getInfo(args.source)
    print("Source Info: ", info)

if args.command == "shrink":
    extractFrames(args.source, args.dest, args.verbose)
elif args.command == "roi":
    points = [int(x) for x in args.rect.split(",")]
    points = group(points, 2)
    extractROI(args.source, args.dest, points, args.verbose)


# didn't work?
# python preprocess_video.py --source /home/ben/VideoUnderstanding/example_images/Animals6mins/Animals6mins.mp4 --dest /home/ben/VideoUnderstanding/example_images/Animals6mins/ --command "shrink"
