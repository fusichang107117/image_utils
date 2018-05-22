#!/usr/bin/python
# -*- coding: UTF-8 -*-
# import tensorflow as tf
import glob
import os
import re
import sys
import cv2
import numpy as np

def video_label_gen(anno_path):
    f_in = open(anno_path,"r")
    line = f_in.readline()  # 调用文件的 readline()方法
    video_str_list = []
    video_label = 0
    while line:
        img_name = line.strip('\n')
        box_num = int(f_in.readline().strip("\n"))
        video_label = 0.0
        video_name = img_name.split("/")[0]
        for i in range(box_num):
            box = f_in.readline().strip("\n").split(" ")
            x1,y1,w,h,s = box
            if float(s) > video_label:
                video_label = float(s)
        video_str_list.append(video_name + " " + str(video_label))
        # print(img_name)
        # if not os.path.exists(img_out_dir):
        #     os.makedirs(img_out_dir)
        # cv2.imwrite(os.path.join(img_out_dir,img_name+".jpg"),img)

        line = f_in.readline()

    return  video_str_list

if __name__ == '__main__':
    anno_path = sys.argv[1]
    result_path = sys.argv[2]

    video_str_list = video_label_gen(anno_path)
    print(video_str_list)
    tmp = ""
    tmp_line = video_str_list[0]
    print(tmp_line)
    name_1st, label_1st = tmp_line.split(" ")
    tmp_label = float(label_1st)

    print(len(video_str_list))
    for idx in range(1, len(video_str_list)):
        print(idx)
        print(video_str_list[idx])
        name_n, label_n = video_str_list[idx].split(" ")
        if name_1st != name_n:
            tmp += name_1st + " " + str(tmp_label) + "\n"
            tmp_line = video_str_list[idx]
            name_1st, label_1st = tmp_line.split(" ")
            tmp_label = float(label_1st)
        else:
            if float(label_n) > tmp_label:
                tmp_label = float(label_n)

    # add last line
    tmp += name_1st + " " + str(tmp_label) + "\n"

    f_out = open(result_path,"w")
    f_out.writelines(tmp)
    f_out.close()