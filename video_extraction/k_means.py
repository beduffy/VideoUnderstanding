# USAGE
# python color_kmeans.py --image images/jp.png --clusters 3

# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
import os
import json

def kmeans_image_show(image_path, clusters=3):
    pass
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    # ap.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
    # args = vars(ap.parse_args())
    #
    # load the image and convert it from BGR to RGB so that
    # we can display it with matplotlib
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # show our image
    plt.figure()
    plt.axis("off")
    plt.imshow(image)

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster the pixel intensities
    clt = KMeans(n_clusters = clusters)
    clt.fit(image)

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = utils.centroid_histogram(clt)
    percentage_color_list = utils.get_colors_and_percentages(hist, clt.cluster_centers_)

    print percentage_color_list
    print percentage_color_list[0][1][0]

    bar = utils.plot_colors(hist, clt.cluster_centers_)

    # show our color bart
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()

def kmeans_image(image_path, clusters=3):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster the pixel intensities
    clt = KMeans(n_clusters = clusters)
    clt.fit(image)

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = utils.centroid_histogram(clt)
    percentage_color_list = utils.get_colors_and_percentages(hist, clt.cluster_centers_)

    return percentage_color_list

def extract_batch_image_kmeans(directory_name, clusters=3):
    image_directory_path = os.path.join(example_images, directory_name)
    tasks_path =  os.path.join(image_directory_path, 'tasks.txt')

    filenames = []

    json_path = os.path.join(image_directory_path, 'result_struct.json')

    with open(tasks_path) as fp:
        print 'Opening ' + tasks_path
        for line in fp:
            filenames.append(line[:-1])

    json_struct = {'images' : []}
    for image_name in filenames:
        image_path = os.path.join(image_directory_path, image_name)
        percentage_color_info = kmeans_image(image_path)
        print percentage_color_info
        json_struct['images'].append({'name' : image_name, 'kmeans': percentage_color_info})

    print json_struct
    json.dump(json_struct, open(json_path, 'w'))


current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
example_images = os.path.join(parent_dir, 'example_images')

# extract_batch_image_kmeans('DogsBabies5mins')
extract_batch_image_kmeans('old_example_images')

