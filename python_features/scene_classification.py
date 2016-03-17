import numpy as np
import os
import matplotlib.pyplot as plt
import json

caffe_root = '/home/ben/Downloads/caffe-master/'
import caffe

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
example_images = os.path.join(parent_dir, 'example_images')

def batch_scene_classification(video_path, models):
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    caffe.set_mode_cpu()

    net0 = caffe.Net(models[0]['prototxt'], models[0]['caffemodel'], caffe.TEST)
    net1 = caffe.Net(models[1]['prototxt'], models[1]['caffemodel'], caffe.TEST)

    directory = os.path.dirname(video_path)
    image_directory_path = os.path.join(directory, 'images', 'full')
    json_struct_path = os.path.join(directory, 'metadata', 'result_struct.json')

    json_struct = {}
    if os.path.isfile(json_struct_path):
        with open(json_struct_path) as data_file:
            json_struct = json.load(data_file)

    num_images = len(json_struct['images'])
    for idx, image in enumerate(json_struct['images']):
        image_path = os.path.join(image_directory_path, image['image_name'])
        # print 'scene %d/%d' % (idx. num_images) TODO CRASHES HERE AttributeError: 'int' object has no attribute 'num_images'
        results1 = classify_scene(net0, image_path)
        results2 = classify_scene(net1, image_path)
        json_struct['images'][idx]['scene_results'] = []  #todo ????
        json_struct['images'][idx]['scene_results'] = {'scene_results1' : results1, 'scene_results2' : results2}

    json.dump(json_struct, open(json_struct_path, 'w'))

def classify_scene(net, image_path):
    # TODO PROCESS ALL IMAGES FIRST?
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    # set net to batch size of 50
    # net.blobs['data'].reshape(50,3,227,227)

    # todo dot dot dot operator?????
    #net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'examples/images/cat.jpg'))
    # net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('/home/ben/VideoUnderstanding/example_images/test_dataset/park2.jpg'))
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image_path))
    out = net.forward()
    # print("Predicted class is #{}.".format(out['prob'][0].argmax()))

    plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))

    # load labels
    # imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
    imagenet_labels_filename = '/home/ben/Downloads/placesCNN/categoryIndex_places205.csv'
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

    # sort top k predictions from softmax output
    top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    # print 'xxxxx', net.blobs['prob'].data[0]
    # print 'xxxxx', net.blobs['prob'].data[0].flatten()
    # print 'xxxxx', net.blobs['prob'].data[0].flatten().argsort()[::-1]
    # print 'xxxxx', labels[top_k]

    top_results = []

    for index in top_k:
        # print labels[index], net.blobs['prob'].data[0].flatten()[index]
        top_results.append({'label' : labels[index], 'probability' : str(net.blobs['prob'].data[0].flatten()[index])})

    return top_results

def main_scene_classification(json_struct, video_path):
    places205model = {'prototxt' : '/home/ben/Downloads/placesCNN/places205CNN_deploy.prototxt', 'caffemodel' : '/home/ben/Downloads/placesCNN/places205CNN_iter_300000.caffemodel'}
    googlenet205model = {'prototxt' : '/home/ben/VideoUnderstanding/python_features/deploy_places205.protxt', 'caffemodel' : '/home/ben/VideoUnderstanding/python_features/googlelet_places205_train_iter_2400000.caffemodel'}

    all_scene_models = []
    all_scene_models.append(places205model)
    all_scene_models.append(googlenet205model)

    batch_scene_classification(video_path, all_scene_models)

# batch_scene_classification('old_example_images', places205model['prototxt'], places205model['caffemodel'])

# batch_scene_classification('old_example_images', '/home/ben/Downloads/placesCNN/places205CNN_deploy.prototxt', '/home/ben/Downloads/placesCNN/places205CNN_iter_300000.caffemodel')















# todo get timeit working and find out why exclamation point doesnt work
# CPU mode
# net.forward()  # call once for allocation
# %timeit net.forward()
# GPU mode
# caffe.set_device(0)
# caffe.set_mode_gpu()
# net.forward()  # call once for allocation
# %timeit net.forward()

# print [(k, v.data.shape) for k, v in net.blobs.items()]
# print [(k, v[0].data.shape) for k, v in net.params.items()]

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.figure()
    plt.imshow(data)
    plt.show()

# the parameters are a list of [weights, biases]
# filters = net.params['conv1'][0].data
# vis_square(filters.transpose(0, 2, 3, 1))
#
# feat = net.blobs['conv1'].data[0, :36]
# vis_square(feat, padval=1)
#
# filters = net.params['conv2'][0].data
# vis_square(filters[:48].reshape(48**2, 5, 5))
#
# feat = net.blobs['conv2'].data[0, :36]
# vis_square(feat, padval=1)
#
# feat = net.blobs['conv3'].data[0]
# vis_square(feat, padval=0.5)
#
# feat = net.blobs['conv4'].data[0]
# vis_square(feat, padval=0.5)
#
# feat = net.blobs['conv5'].data[0]
# vis_square(feat, padval=0.5)
#
# feat = net.blobs['pool5'].data[0]
# vis_square(feat, padval=1)
#
# feat = net.blobs['fc6'].data[0]
# plt.subplot(2, 1, 1)
# plt.plot(feat.flat)
# plt.subplot(2, 1, 2)
# _ = plt.hist(feat.flat[feat.flat > 0], bins=100)
# plt.show()
#
# feat = net.blobs['fc7'].data[0]
# plt.subplot(2, 1, 1)
# plt.plot(feat.flat)
# plt.subplot(2, 1, 2)
# _ = plt.hist(feat.flat[feat.flat > 0], bins=100)
# plt.show()
#
# feat = net.blobs['prob'].data[0]
# plt.plot(feat.flat)
# plt.show()
