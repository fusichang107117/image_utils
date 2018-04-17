#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import glob
import os
import re
import sys
import cv2
import numpy as np

def draw_boxes(anno_path, img_dir) :
    f_in = open(anno_path,"r")
    line = f_in.readline()  # 调用文件的 readline()方法
    while line:
        img_name = line.strip('\n')
        img_path = os.path.join(img_dir,img_name+".jpg")
        img = cv2.imread(img_path)
        box_num = int(f_in.readline().strip("\n"))
        for i in range(box_num):
            box = f_in.readline().strip("\n").split(" ")
            x1 = int(float(box[0]))
            y1 = int(float(box[1]))
            x2 = x1 + int(float(box[2]))
            y2 = y1 + int(float(box[3]))
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0))
        cv2.imshow("test",img)
        cv2.waitKey(0)
        # print(img_name)
        # if not os.path.exists(img_out_dir):
        #     os.makedirs(img_out_dir)
        # cv2.imwrite(os.path.join(img_out_dir,img_name+".jpg"),img)

        line = f_in.readline()

if __name__ == '__main__':
    anno_path = sys.argv[1]
    img_dir = sys.argv[2]
    draw_boxes(anno_path, img_dir)