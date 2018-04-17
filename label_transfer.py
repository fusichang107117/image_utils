import os
import glob
import cv2
import sys

def label_person_resize(label_input, label_output, image_size):

    label_org = open(label_input, "r")
    label_lines = label_org.readlines()

    label_new = open(label_output, "w")

    idx = 0
    label_lines_new = ""
    while (idx < len(label_lines)):
        new_name = label_lines[idx]
        label_lines_new += new_name
        idx+=1
        new_num = label_lines[idx]
        label_lines_new += new_num
        idx+=1
        for i in range(0, int(new_num)):
            x0,y0,w,h = label_lines[idx].strip("\n").split(" ")
            print(x0,y0,w,h)
            print(str(int(float(x0)/1920*image_size[0])) + " " +
                  str(int(float(y0)/1080*image_size[1])) + " " +
                  str(int(float(w)/1920*image_size[0])) + " " +
                  str(int(float(h)/1080*image_size[1]))
                  )
            x1 = int(x0) + int(w)
            y1 = int(y0) + int(h)
            x0_new = int(float(x0)/1920*image_size[0])
            y0_new = int(float(y0)/1080*image_size[1])
            x1_new = int(float(x1)/1920*image_size[0])
            y1_new = int(float(y1)/1080*image_size[1])
            w_new = x1_new - x0_new
            h_new = y1_new - y0_new

            label_lines_new += (str(x0_new) + " " +
                                str(y0_new) + " " +
                                str(w_new) + " " +
                                str(h_new) + "\n"
                                )
            idx+=1

    label_new.writelines(label_lines_new)
    label_new.close()
    label_org.close()

def label_resize(label_input, label_output, image_size):

    label_org = open(label_input, "r")
    label_lines = label_org.readlines()

    label_new = open(label_output, "w")

    idx = 0
    label_lines_new = ""
    while (idx < len(label_lines)):
        new_name = label_lines[idx]
        label_lines_new += new_name
        idx+=1
        new_num = label_lines[idx]
        label_lines_new += new_num
        idx+=1
        for i in range(0, int(new_num)):
            x0,y0,w,h,score = label_lines[idx].strip("\n").split(" ")
            print(x0,y0,w,h,score)
            print(str(int(float(x0)/1920*image_size[0])) + " " +
                  str(int(float(y0)/1080*image_size[1])) + " " +
                  str(int(float(w)/1920*image_size[0])) + " " +
                  str(int(float(h)/1080*image_size[1])) + " " +
                  score)
            x1 = int(x0) + int(w)
            y1 = int(y0) + int(h)
            x0_new = int(float(x0)/1920*image_size[0])
            y0_new = int(float(y0)/1080*image_size[1])
            x1_new = int(float(x1)/1920*image_size[0])
            y1_new = int(float(y1)/1080*image_size[1])
            w_new = x1_new - x0_new
            h_new = y1_new - y0_new

            label_lines_new += (str(x0_new) + " " +
                                str(y0_new) + " " +
                                str(w_new) + " " +
                                str(h_new) + " " +
                                score + "\n"
                                )
            idx+=1

    label_new.writelines(label_lines_new)
    label_new.close()
    label_org.close()

if __name__ == '__main__':
    label_input = sys.argv[1]
    label_output = sys.argv[2]


    label_resize(label_input, label_output, (1280, 720))