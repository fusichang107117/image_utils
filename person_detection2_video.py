
# coding: utf-8

# In[ ]:

# # Imports

# In[ ]:

import numpy as np
import os
import cv2
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import copy

from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


# ## Env setup

# In[ ]:


# ## Object detection imports
# Here are the imports from the object detection module.

# In[ ]:

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

from object_detection.utils import read_mp4


# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[ ]:

# What model to download.

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = "/home/lqy/faster_rcnn_person.pb"

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = "/home/lqy/models/research/object_detection/data/mscoco_label_map.pbtxt"

NUM_CLASSES = 1


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[ ]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[ ]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[ ]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
IMAGES_DIR = "/home/lqy/contribution/20180418"
IMAGE_SUFFIX = ".mp4"
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
# IMAGE_SIZE = (12, 8)
IMAGE_LIST = "/home/lqy/video_list_3000_3000.txt"
DETECTION_RESULT_PATH = "/home/lqy/video_person_detection_result_faster_rcnn.txt"

list = open(IMAGE_LIST, "r")
TEST_IMAGE_PATHS = list.readlines()
list.close()

# In[ ]:

def run_inference_for_single_image(images, sess, graph, tensor_dict, image_tensor):
  # with graph.as_default():
  #   with tf.Session() as sess:
  #     # Get handles to input and output tensors
  #     ops = tf.get_default_graph().get_operations()
  #     all_tensor_names = {output.name for op in ops for output in op.outputs}
  #     tensor_dict = {}
  #     for key in [
  #         'num_detections', 'detection_boxes', 'detection_scores',
  #         'detection_classes', 'detection_masks'
  #     ]:
  #       tensor_name = key + ':0'
  #       if tensor_name in all_tensor_names:
  #         tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
  #             tensor_name)
  #     if 'detection_masks' in tensor_dict:
  #       # The following processing is only for single image
  #       detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
  #       detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
  #       # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
  #       real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
  #       detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
  #       detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
  #       detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
  #           detection_masks, detection_boxes, image.shape[0], image.shape[1])
  #       detection_masks_reframed = tf.cast(
  #           tf.greater(detection_masks_reframed, 0.5), tf.uint8)
  #       # Follow the convention by adding back the batch dimension
  #       tensor_dict['detection_masks'] = tf.expand_dims(
  #           detection_masks_reframed, 0)
  #     image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

  for index, img in enumerate(images):
      image = copy.deepcopy(img[:, :, ::-1])

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
      # print(output_dict['num_detections'])
      # print(output_dict['detection_classes'])
      # print(output_dict["detection_scores"])
      idx = np.where(output_dict['detection_classes'] == 1)
      print(idx)
      dims = idx
      # dims = np.where(output_dict["detection_scores"][idx_set]>0.1)
      # print(np.shape(dims))
      # print(dims[0])
      output_dict['num_detections'] = len(dims[0])
      output_dict['detection_boxes'] = output_dict['detection_boxes'][dims[0]]
      output_dict['detection_scores'] = output_dict['detection_scores'][dims[0]]
      output_dict['detection_classes'] = output_dict['detection_classes'][dims[0]]
      # print(output_dict)
      print(output_dict["detection_scores"])

      dims = np.where(output_dict["detection_scores"] >= 0.5)
      # # print(np.shape(dims))
      # h,w,c = np.shape(image)
      # # print(h,w,c)
      # str_=""
      # num = str(len(dims[0])) + "\n"
      # str_+=num
      # for idx in dims[0]:
      #    #print(output_dict["detection_boxes"][idx])
      #    ymin = int(output_dict["detection_boxes"][idx][0]*h)
      #    xmin = int(output_dict["detection_boxes"][idx][1]*w)
      #    ymax = int(output_dict["detection_boxes"][idx][2]*h)
      #    xmax = int(output_dict["detection_boxes"][idx][3]*w)
      #    b_w  = xmax - xmin
      #    b_h  = ymax - ymin
      #    str_+= str(xmin) + " " + str(ymin) + " " + str(b_w) + " " + str(b_h) + " " + str(output_dict["detection_scores"][idx]) + "\n"

      # print(str_)

      print("idx = {}, objs_num = {}".format(index, len(dims[0])))
      if len(dims[0]) > 0:
          break

  return len(dims[0])


with detection_graph.as_default():
  with tf.Session() as sess:
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
            tensor_name)
    # if 'detection_masks' in tensor_dict:
    #   # The following processing is only for single image
    #   detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
    #   detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
    #   # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    #   real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    #   detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    #   detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    #   detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
    #       detection_masks, detection_boxes, image.shape[0], image.shape[1])
    #   detection_masks_reframed = tf.cast(
    #       tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    #   # Follow the convention by adding back the batch dimension
    #   tensor_dict['detection_masks'] = tf.expand_dims(
    #       detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # In[ ]:
    result_str=""
    for image_path in TEST_IMAGE_PATHS:

        image_full_path = os.path.join(IMAGES_DIR, image_path.strip("\n") + IMAGE_SUFFIX)
        # load image
        images = read_mp4.get_frames_by_second(image_full_path, 0, 5, 1, 640, 360)
        print(len(images))
        name = image_path.strip("\n")
        flag = 0

        print("processing {}".format(name))
        if len(images) > 0:
            obj_num = run_inference_for_single_image(images, sess, detection_graph, tensor_dict, image_tensor)
            if obj_num > 0:
                flag = 1


        result_str += image_path.strip("\n") + " " + str(flag) + "\n"




result_score = open(DETECTION_RESULT_PATH, "w")
result_score.writelines(result_str)
result_score.close()
# In[ ]:
