import numpy as np
import cv2

def calculate_histograms(image, bins=(8, 12, 3)):
    # compute a 3D histogram in the RGB colorspace,
    # then normalize the histogram so that images
    # with the same content, but either scaled larger
    # or smaller will have (roughly) the same histogram
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    features = []

    (h, w) = image.shape[:2]
    (cX, cY) = (int(w * 0.5), int(h * 0.5))

    # divide the image into four rectangles/segments (top-left,
    # top-right, bottom-right, bottom-left)
    segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
        (0, cX, cY, h)]

    # construct an elliptical mask representing the center of the
    # image
    (axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
    ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
    cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

    # loop over the segments
    for (startX, endX, startY, endY) in segments:
        # construct a mask for each corner of the image, subtracting
        # the elliptical center from it
        cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
        cornerMask = cv2.subtract(cornerMask, ellipMask)

        # extract a color histogram from the image, then update the
        # feature vector
        hist = histogram(image, cornerMask, bins)
        features.extend(hist)

    # extract a color histogram from the elliptical region and
    # update the feature vector
    hist = histogram(image, ellipMask, bins)
    features.extend(hist)

    # print features

    # return out 3D histogram as a flattened array
    return features

def histogram(image, mask, bins):
    # extract a 3D color histogram from the masked region of the
    # image, using the supplied number of bins per channel; then
    # normalize the histogram
    hist = cv2.calcHist([image], [0, 1, 2], mask, bins,
        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist).flatten()

    # return the histogram
    return hist


def chi2_distance(histA, histB, eps = 1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
        for (a, b) in zip(histA, histB)])

    # return the chi-squared distance
    return d

def chi2_distance_two_images(image1, image2):
    hist1 = calculate_histograms(image1)
    hist2 = calculate_histograms(image2)
    return chi2_distance(hist1, hist2)
