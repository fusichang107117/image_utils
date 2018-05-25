import os
import random
import sys
import numpy as np

CKimg_dir = "JPEGImages"
CKanno_dir = "Annotations"

def mkr(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# split train and test for training
def split_traintest(voc_path, trainratio=0.8, valratio=0.1, testratio=0.1):
    dataset_dir = voc_path
    files = os.listdir(os.path.join(voc_path, CKimg_dir))
    trains = []
    vals = []
    trainvals = []
    tests = []
    random.shuffle(files)
    for i in range(len(files)):
        filepath = os.path.join(voc_path, CKimg_dir) + "/" + files[i][:-3] + "jpg"
        if (i < trainratio * len(files)):
            trains.append(files[i])
            trainvals.append(files[i])
        elif i < (trainratio + valratio) * len(files):
            vals.append(files[i])
            trainvals.append(files[i])
        else:
            tests.append(files[i])
    # uncomment yolo
    #         # write txt files for yolo
    # with open(dataset_dir + "/trainval.txt", "w")as f:
    #     for line in trainvals:
    #         line = CKimg_dir + "/" + line
    #         f.write(line + "\n")
    # with open(dataset_dir + "/test.txt", "w") as f:
    #     for line in tests:
    #         line = CKimg_dir + "/" + line
    #         f.write(line + "\n")
            # write files for voc
    maindir = dataset_dir + "/" + "ImageSets/Main"
    mkr(maindir)
    with open(maindir + "/train.txt", "w") as f:
        for line in trains:
            line = line[:line.rfind(".")]
            f.write(line + "\n")
    with open(maindir + "/val.txt", "w") as f:
        for line in vals:
            line = line[:line.rfind(".")]
            f.write(line + "\n")
    with open(maindir + "/trainval.txt", "w") as f:
        for line in trainvals:
            line = line[:line.rfind(".")]
            f.write(line + "\n")
    with open(maindir + "/test.txt", "w") as f:
        for line in tests:
            line = line[:line.rfind(".")]
            f.write(line + "\n")
    print("spliting done")


if __name__ == "__main__":
    voc_path = sys.argv[1]
    split_traintest(voc_path)