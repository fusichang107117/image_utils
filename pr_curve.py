import glob
import os
import re
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def cal_pr_point(gt_l, rel_l, thr=0.5):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)

    gt_len = len(gt_l)
    rel_len = len(rel_l)

    comp_len = min(gt_len, rel_len)

    for i in range(0, comp_len):
        gt_name, gt_score = gt_l[i].split(" ")
        rel_name, rel_score = rel_l[i].split(" ")
        if gt_name != rel_name:
            print("file name mismatch {}-{}".format(gt_name, rel_name))
        else:
            if float(rel_score) >= thr and float(gt_score) >= 1.0:
                tp += 1
            elif float(rel_score) < thr and float(gt_score) <= 0.0:
                tn += 1
            elif float(rel_score) >= thr and float(gt_score) <= 0.0:
                fp += 1
            elif float(rel_score) < thr and float(gt_score) >= 1.0:
                fn += 1

    print("thr={}, tp={}, fp={}, tn={}, fn={}".format(thr, tp, fp, tn, fn))

    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)

    return precision, recall

if __name__ == '__main__':
    anno_path = sys.argv[1]
    result_path = sys.argv[2]

    anno_fd = open(anno_path, "r")
    anno_lines = anno_fd.readlines()
    anno_fd.close()

    anno_list = []
    for line in anno_lines:
        anno_list.append(line.strip("\n"))


    rel_fd = open(result_path, "r")
    rel_lines = rel_fd.readlines()
    rel_fd.close()

    rel_list = []
    for line in rel_lines:
        rel_list.append(line.strip("\n"))

    p_point=[]
    r_point=[]
    for th in np.arange(0.95, -0.05, -0.05):
        p,r = cal_pr_point(anno_list, rel_list, th)
        p_point.append(p)
        r_point.append(r)


    for i in range(0, len(p_point)):
        print(p_point[i], r_point[i])

    x0 = r_point[6]
    y0 = p_point[6]
    plt.plot(r_point, p_point)

    plt.plot([x0,x0] ,[0,y0]  ,'k--')
    plt.plot([x0, 0] ,[y0,y0] ,'k--')
    plt.scatter(x0, y0, s=20, c="r")
    plt.text(x0, y0, str("({},{})".format(round(x0,2),round(y0,2))))

    x = np.arange(0.0, 1.1, 0.1)
    y = np.arange(0.0, 1.1, 0.1)
    plt.xticks(x)
    plt.yticks(y)
    plt.title('PR-Curve')
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title(os.path.basename(result_path))
    plt.show()