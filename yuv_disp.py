import os
import sys
import cv2
import numpy as np



def yuv_disp(path, h, w):
    fd_raw = open(path, "rb")
    if fd_raw == None:
        print("invalid path {}".format(path))
        return
    print(h,w)
    disp_img = np.zeros((h*w, 1), dtype=np.uint8)
    raw_data = fd_raw.read(h*w)
    while(raw_data != None):
        for i in range(h*w):
            disp_img[i] = ord(raw_data[i])
        print("show image")
        cv2.imshow("Y", disp_img.reshape(h,w))
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        raw_data = fd_raw.read(h * w)

def yuv_disp2(path, h, w):
    fsize = os.path.getsize(path)
    frm_cnt = fsize/(h*w)

    fd_raw = open(path, "rb")
    if fd_raw == None:
        print("invalid path {}".format(path))
        return
    print(h,w)
    disp_img = np.zeros((h*w, 1), dtype=np.uint8)

    idx = 1
    raw_data = fd_raw.read(h*w)
    while(raw_data != None):
        fd_raw.seek(idx*(h*w), 0)
        raw_data = fd_raw.read(h*w)
        for i in range(h*w):
            disp_img[i] = ord(raw_data[i])
        print("show image")
        cv2.imshow("Y", disp_img.reshape(h,w))
        key = cv2.waitKey(0)
        # up or left
        print(key)
        if key == 82 or key == 81:
            idx -= 1
        # down or right
        elif key == 84 or key == 83:
            idx += 1
        # enter
        elif key == 13:
            idx += 1
        elif key == ord('q'):
            break

        if idx < 0:
            idx = 0



if __name__ == "__main__":
    yuv_disp2(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

