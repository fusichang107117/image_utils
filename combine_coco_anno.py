from pycocotools.coco import COCO
import numpy as np
import json

src_paths = ['/home/leon/Images/coco/annotations_2017_reduced/instances_train2017_reduced.json',
             '/home/leon/Images/coco/annotations_2017_reduced/instances_val2017_reduced.json']
dst_path = '/home/leon/Images/coco/annotations_2017_reduced/instances_trainval2017_reduced.json'

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
    catIds = coco.getCatIds(catNms=['person'])
    print catIds
    imgIds = coco.getImgIds(catIds=catIds)
    print len(imgIds)
    # print imgIds
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

    # load and display instance annotations
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    print annIds
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    print "----------------------"

def combine_coco_anno(src_anno_paths, dst_anno_path):
    dst_coco_json = []
    first = True
    for src_anno_path in src_anno_paths:
        coco_file = open(src_anno_path,"r")
        coco_json = json.load(coco_file)
        if  first:
            dst_coco_json = coco_json
            first = False
        dst_coco_json["images"].extend(coco_json["images"])
        dst_coco_json["annotations"].extend(coco_json["annotations"])
        coco_file.close()

    json_file = open(dst_anno_path, "w")
    json.dump(dst_coco_json, json_file)
    json_file.close()


if __name__ == '__main__' :
    # combine_coco_anno(src_paths, dst_path)
    test_coco_api("/home/leon/Images/PASCAL_VOC/pascal_train2012.json")
    # test_coco_api("/home/leon/Images/coco/annotations_2017_reduced/instances_val2017_reduced.json")
    # test_coco_api(dst_path)