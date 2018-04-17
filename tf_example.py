import tensorflow as tf
import glob
import os
import re
import cv2
import numpy as np

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_path', '', 'Path to image')
flags.DEFINE_string('image_list', '', 'Path to image')
FLAGS = flags.FLAGS


def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  height = example["height"]                # Image height
  width = example["width"]                  # Image width
  filename = example["filename"]            # Filename of the image. Empty if image is not from file
  encoded_image_data = example["encoded"]   # Encoded image bytes
  image_format = example["format"]          # b'jpeg' or b'png'

  xmins = example["xmin"]                   # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = example["xmax"]                   # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = example["ymin"]                   # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = example["ymax"]                   # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = example["class_text"]      # List of string class name of bounding box (1 per box)
  classes = example["class_label"]          # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/filename': bytes_feature(filename),
      'image/source_id': bytes_feature(filename),
      'image/encoded': bytes_feature(encoded_image_data),
      'image/format': bytes_feature(image_format),
      'image/object/bbox/xmin': float_list_feature(xmins),
      'image/object/bbox/xmax': float_list_feature(xmaxs),
      'image/object/bbox/ymin': float_list_feature(ymins),
      'image/object/bbox/ymax': float_list_feature(ymaxs),
      'image/object/class/text': bytes_list_feature(classes_text),
      'image/object/class/label': int64_list_feature(classes),
  }))
  return tf_example


def main(_):

    if not os.path.exists(FLAGS.image_list) or not os.path.exists(FLAGS.image_path):
        print("path doesn't exist")
        return

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    imgs_list = open(FLAGS.image_list,"r")
    lines = imgs_list.readlines()

    count = 0
    for line in lines:
        example ={}
        print(line)
        img_name, bboxes = line.split(":")
        img_name = img_name.strip("u'{")
        # img_name = img_name.split("/")[-1]
        # print(img_name)
        # print(bboxes)
        bboxes = re.sub('[\[,\]}]', "", bboxes)
        bboxes = bboxes.split()
        # print(bboxes)
        bboxes = map(np.float, bboxes)
        bboxes = np.array(bboxes).reshape(-1,4)
        # print(bboxes)
        xmin=bboxes[:,0]
        ymin=bboxes[:,1]
        xmax=bboxes[:,2]
        ymax=bboxes[:,3]
        # print(xmin)
        # print(ymin)
        # print(xmax)
        # print(ymax)
        person_num = np.shape(bboxes)[0]
        class_label = np.ones(person_num, np.int32)
        class_text = ['person'] * person_num
        # print(class_label)
        # print(class_text)
        img_path = os.path.join(FLAGS.image_path, img_name)
        img_data = tf.gfile.FastGFile(img_path, "r").read()
        img = cv2.imread(img_path)
        height, width, channel = np.shape(img)
        # print(height, width, channel)

        # write example into tfrecords
        example["height"] = height
        example["width"] = width
        example["channel"] = channel
        example["filename"] = img_name
        example["source_id"] = img_name
        example["encoded"] = img_data
        example["format"] = b'jpeg'
        example["xmin"] = xmin/float(width)
        example["ymin"] = ymin/float(height)
        example["xmax"] = xmax/float(width)
        example["ymax"] = ymax/float(height)
        example["class_label"] = class_label
        example["class_text"] = class_text

        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())
        # check ground truth boxes
        # for box in bboxes:
        #     cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 2)
        #
        # cv2.imshow("display", img)
        # cv2.waitKey(0)

        # for bbox in bboxes:
        # count+=1
        # if count == 2:
        #     exit()

    writer.close()


if __name__ == '__main__':
  tf.app.run()