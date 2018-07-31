import cv2
import sys
import os
import numpy as np

img_dir = "./xiaomi_faces_align_160_val"
img_pair = "face_list.txt"


pair_fd = open(img_pair, "r")
pair_lines = pair_fd.readlines()

for l in pair_lines:
    lp, rp = l.strip("\n").split(" ")
    print(lp, rp)
    lp_full = os.path.join(img_dir, lp)
    rp_full = os.path.join(img_dir, rp)
    img_left = cv2.imread(lp_full)
    img_right = cv2.imread(rp_full)
    img_left_scale = cv2.resize(img_left, (128,128))
    img_right_scale = cv2.resize(img_right, (128, 128))

    #img_right = img_right[np.newaxis, :,:,:]
    img_disp = np.concatenate((img_left_scale, img_right_scale), axis=1)
    cv2.imshow("pair_disp", img_disp)
    cv2.waitKey(0)