import numpy as np
import cv2


def calculate_hist(image, bins=(8, 8, 8)):
    # compute a 3D histogram in the RGB colorspace,
    # then normalize the histogram so that images
    # with the same content, but either scaled larger
    # or smaller will have (roughly) the same histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist)

    # return out 3D histogram as a flattened array
    return hist.flatten()

def chi2_distance(histA, histB, eps = 1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
        for (a, b) in zip(histA, histB)])

    # return the chi-squared distance
    return d

def chi2_distance_two_images(image1, image2):
    hist1 = calculate_hist(image1)
    hist2 = calculate_hist(image2)
    return chi2_distance(hist1, hist2)
#
# class Searcher:
#     def __init__(self, index):
# 		# store our index of images
# 		self.index = index
#
# 	def search(self, queryFeatures):
# 		# initialize our dictionary of results
# 		results = {}
#
# 		# loop over the index
# 		for (k, features) in self.index.items():
# 			# compute the chi-squared distance between the features
# 			# in our index and our query features -- using the
# 			# chi-squared distance which is normally used in the
# 			# computer vision field to compare histograms
# 			d = self.chi2_distance(features, queryFeatures)
#
# 			# now that we have the distance between the two feature
# 			# vectors, we can udpate the results dictionary -- the
# 			# key is the current image ID in the index and the
# 			# value is the distance we just computed, representing
# 			# how 'similar' the image in the index is to our query
# 			results[k] = d
#
# 		# sort our results, so that the smaller distances (i.e. the
# 		# more relevant images are at the front of the list)
# 		results = sorted([(v, k) for (k, v) in results.items()])
#
# 		# return our results
# 		return results
