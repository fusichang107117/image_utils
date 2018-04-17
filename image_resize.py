import os
import glob
import cv2
import sys

dir_input  = "/home/leon/AI/EvaluationSetPro/face/reg/xiaomi_faces_from_1920_1080_margin_22/"
dir_output = "/home/leon/AI/EvaluationSetPro/face/reg/xiaomi_faces_160_from_1920_1080_margin_22/"



def create_image_list(dir_input):
    image_list = []

    # add sub dir
    img_dirs = os.listdir(dir_input)
    # print(img_dirs)
    for img_dir in img_dirs:
        imgs_list = glob.glob(os.path.join(os.path.join(dir_input,img_dir), "*.jpg"))
        image_list += imgs_list

    # add jpg file
    img_dirs = glob.glob(os.path.join(dir_input, "*.jpg"))
    image_list += img_dirs

    print(image_list)

    return image_list
# print(image_list)
# exit()
def image_resize(dir_output, image_list, image_size):
    for img_path in image_list:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size[0], image_size[1]))
        print(img_path.split("/"))
        sub_dir = img_path.split("/")[-2]
        full_dir = os.path.join(dir_output, sub_dir)
        if not os.path.exists(full_dir):
            os.makedirs(full_dir)
        cv2.imwrite(os.path.join(full_dir, os.path.basename(img_path)), img)

if __name__ == '__main__':
    dir_input = sys.argv[1]
    dir_output = sys.argv[2]

    if not os.path.exists(dir_input):
        print("invalid input dir")
        exit()

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    image_list = create_image_list(dir_input)
    image_resize(dir_output, image_list, (640, 360))
