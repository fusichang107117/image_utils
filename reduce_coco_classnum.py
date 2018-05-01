from pycocotools.coco import COCO
import numpy as np
import json
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

ann_train_file = '/home/leon/Images/coco/annotations_2014/instances_val2014.json'
ann_train_file_reduced = '/home/leon/Images/coco/annotations_2014/instances_val2014_reduced.json'
need_ids = {1,2,3,4,6,8,16,17,18}

def test_coco_api(anno_path):
    # coco_train = COCO(ann_train_file)
    # print len(coco_train.dataset['categories'])
    # print coco_train.dataset['categories']
    # initialize COCO api for instance annotations
    coco = COCO(anno_path)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['person', 'dog', 'skateboard'])
    print catIds
    imgIds = coco.getImgIds(catIds=catIds)
    print imgIds
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

    # load and display instance annotations
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    print annIds
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)

def is_need_id(id):
    for need_id in need_ids:
        if id == need_id:
            return True
    return False

def convert_id(old_id):
    for i, need_id in enumerate(need_ids):
        if old_id == need_id:
            return i+1
    return -1

def reduce_classnum():
    json_file = open(ann_train_file,"r")
    json_coco = json.load(json_file)
    json_file.close()
    print json_coco["info"]
    print json_coco["licenses"]
    print "images nums: %d, [0]: %s" % (len(json_coco["images"]),json_coco["images"][0])
    print "annotations nums: %d, [0]: %s" % (len(json_coco["annotations"]), json_coco["annotations"][0])
    print json_coco["categories"]

    new_json_anno = []
    for anno in json_coco["annotations"]:
        if is_need_id(anno["category_id"]):
            anno["category_id"] = convert_id(anno["category_id"])
            new_json_anno.append(anno)
    print "new annotations nums: %d, [0]: %s" % (len(new_json_anno), new_json_anno[0])

    new_json_cate= []
    for cate in json_coco["categories"]:
        if is_need_id(cate["id"]):
            cate["id"] = convert_id(cate["id"])
            new_json_cate.append(cate)
    print "new categories : %s" % new_json_cate

    json_coco["annotations"] = new_json_anno
    json_coco["categories"] = new_json_cate

    print "annotations nums: %d, [0]: %s" % (len(json_coco["annotations"]), json_coco["annotations"][0])
    print json_coco["categories"]

    json_file = open(ann_train_file_reduced, "w")
    json.dump(json_coco, json_file)
    json_file.close()


if __name__ == '__main__' :
    reduce_classnum()
    test_coco_api(ann_train_file_reduced)