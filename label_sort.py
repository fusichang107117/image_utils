#!/usr/bin/python
# -*- coding: UTF-8 -*-
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
        video_label = box_num
        video_name = img_name.split("/")[0]
        for i in range(box_num):
            box = f_in.readline().strip("\n").split(" ")

        video_str_list.append(video_name + " " + str(video_label))
        # print(img_name)
        # if not os.path.exists(img_out_dir):
        #     os.makedirs(img_out_dir)
        # cv2.imwrite(os.path.join(img_out_dir,img_name+".jpg"),img)

        line = f_in.readline()

    return  video_str_list

if __name__ == '__main__':
    list_path = sys.argv[1]
    result_path = sys.argv[2]

    result_sort_path = os.path.join(os.path.dirname(result_path), os.path.basename(result_path).split(".")[0] + "_sort.txt")
    print("result_sort: {}".format(result_sort_path))

    list_fd = open(list_path, "r")
    dst_lines = list_fd.readlines()

    dst_seq = []
    for line in dst_lines:
        dst_seq.append(line.strip("\n"))

    result_fd = open(result_path, "r")
    result_lines = result_fd.readlines()

    result_seq = []
    result_seq_name = []
    for line in result_lines:
        result_seq.append(line.strip("\n"))
        result_seq_name.append(line.strip("\n").split(" ")[0])

    sort_seq = []

    for seq in dst_seq:
        # add name
        idx = result_seq_name.index(seq)
        sort_seq.append(result_seq[idx] + "\n")
        # # add num
        # num = result_seq[idx + 1]
        # sort_seq.append(num + "\n")
        # # add bbox
        # for j in num:
        #     sort_seq.append(result_seq[idx+1+j] + "\n")

    f_out = open(result_sort_path,"w")
    f_out.writelines(["%s" % seq  for seq in sort_seq])
    f_out.close()