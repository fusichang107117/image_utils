from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import cv2
import shutil
from lxml import etree, objectify
from tqdm import tqdm
import random
import sys
import numpy as np

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

CK5cats = ['person']

minsize2select = 20
CKimg_dir = "JPEGImages"
CKanno_dir = "Annotations"

def mkr(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def showimg(coco, dataDir, dataType, img, CK5Ids):
    # global dataDir
    I = io.imread('%s/%s/%s' % (dataDir, dataType, img['file_name']))
    plt.imshow(I)
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=CK5Ids, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()


def save_annotations(img, voc_path, dataDir, dataType, filename, objs):
    annopath = os.path.join(voc_path, CKanno_dir) + "/" + filename[:-3] + "xml"
    dst_path = os.path.join(voc_path, CKimg_dir) + "/" + filename
    img_path = dataDir + "/" + dataType + "/" + filename
    # img = cv2.imread(img_path)
    if np.shape(img)[2] != 3:
        print(filename + " not a RGB image")
        return
    # shutil.copy(img_path, dst_path)
    cv2.imwrite(dst_path, img)
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('1'),
        E.filename(filename),
        E.source(
            E.database('CKdemo'),
            E.annotation('VOC'),
            E.image('CK')
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(img.shape[2])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose(),
            E.truncated("0"),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[2]),
                E.ymin(obj[3]),
                E.xmax(obj[4]),
                E.ymax(obj[5])
            )
        )
        anno_tree.append(anno_tree2)
    etree.ElementTree(anno_tree).write(annopath, pretty_print=True)


def showbycv(coco, voc_path, dataDir, dataType, img, classes, CK5Ids):
    # global dataDir
    filename = img['file_name']
    filepath = '%s/%s/%s' % (dataDir, dataType, filename)
    I = cv2.imread(filepath)
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=CK5Ids, iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    ann_count = 0
    for ann in anns:
        name = classes[ann['category_id']]
        if name in CK5cats:
            if ann["iscrowd"] == 1:
                print("skip crowd annotation")
                continue
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin = (int)(bbox[0])
                ymin = (int)(bbox[1])
                xmax = (int)(bbox[2] + bbox[0])
                ymax = (int)(bbox[3] + bbox[1])
                if int(bbox[2]) >= minsize2select and int(bbox[3]) >= minsize2select:
                    obj = [name, 1.0, xmin, ymin, xmax, ymax]
                    objs.append(obj)
                    ann_count += 1
                else:
                    I[ymin:ymax, xmin:xmax, :] = (104, 117, 123)
                # cv2.rectangle(I, (xmin, ymin), (xmax, ymax), (255, 0, 0))
                # cv2.putText(I, name, (xmin, ymin), 3, 1, (0, 0, 255))
    if ann_count > 0:
        save_annotations(I, voc_path, dataDir, dataType, filename, objs)
        # cv2.imshow("img", I)
        # cv2.waitKey(1)


def catid2name(coco):
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
        # print(str(cat['id'])+":"+cat['name'])
    return classes


def get_CK5(voc_path, dataDir):

    if not os.path.exists(voc_path):
        os.mkdir(voc_path)
    mkr(os.path.join(voc_path, CKimg_dir))
    mkr(os.path.join(voc_path, CKanno_dir))
    dataTypes = ['train2014', 'val2014']
    for dataType in dataTypes:
        annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
        coco = COCO(annFile)
        CK5Ids = coco.getCatIds(catNms=CK5cats)
        classes = catid2name(coco)
        for srccat in CK5cats:
            print(dataType + ":" + srccat)
            catIds = coco.getCatIds(catNms=[srccat])
            imgIds = coco.getImgIds(catIds=catIds)
            print("{} has {} images".format(srccat, len(imgIds)))
            # imgIds=imgIds[0:100]
            for imgId in tqdm(imgIds):
                img = coco.loadImgs(imgId)[0]
                showbycv(coco, voc_path, dataDir, dataType, img, classes, CK5Ids)
                # showimg(coco,dataType,img,CK5Ids)


# split train and test for training
def split_traintest(voc_path, trainratio=0.9, valratio=0.05, testratio=0.05):
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
    coco_path = sys.argv[2]
    get_CK5(voc_path, coco_path)
    split_traintest(voc_path)