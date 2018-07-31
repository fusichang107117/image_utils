import random
import sys
import os
import argparse
import math


def nCr(n, r):
    if n < r:
        return 0
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def image_db_build(img_folder, select_matched_count, select_mismatched_count):
    classes = os.listdir(img_folder)
    db = []

    total_image_count = 0
    for c in classes:
        path = os.path.join(img_folder, c)
        if not os.path.isdir(path):
            continue
        #files = [os.path.join(img_folder, c, e) for e in os.listdir(os.path.join(img_folder, c))]
        files = [os.path.join(c, e) for e in os.listdir(os.path.join(img_folder, c))]
        db.append(files)
        total_image_count += len(files)

    total_matched_pairs = 0
    total_mismatched_pairs = 0

    for c, files in enumerate(db):
        total_matched_pairs += nCr(len(files), 2)
        total_mismatched_pairs += len(files) * (total_image_count - len(files))

    #print("Number of classes: %d" % len(db), file=sys.stderr)
    #print("Total matched pairs in directory: %d" % total_matched_pairs, file=sys.stderr)
    #print("Total mismatched pairs in directory: %d" % total_mismatched_pairs, file=sys.stderr)
    sys.stderr.write("Number of classes: {}".format(len(db)))
    sys.stderr.write("Total matched pairs in directory: {}".format(total_matched_pairs))
    sys.stderr.write("Total mismatched pairs in directory: {}".format(total_mismatched_pairs))

    if select_matched_count > total_matched_pairs or select_mismatched_count > total_mismatched_pairs:
        print("Invalid selecting count")
        return None
    else:
        return db


def do_select_matched(db, count):
    matched_set = set()
    while len(matched_set) < count:
        idx = random.randrange(0, len(db))
        lst = db[idx]
        f1 = random.choice(lst)
        f2 = random.choice(lst)
        if f1 == f2:
            continue
        pair = tuple([idx]*2 + sorted([f1, f2]))
        matched_set.add(pair)

    for pair in matched_set:
        idx1, idx2, f1, f2 = pair
        #print("%d %d %s %s" % (idx, idx, f1, f2))
        print("%s %s" % (f1, f2))

    return matched_set


def do_select_mismatched(db, count):
    mismatched_set = set()
    while len(mismatched_set) < count:
        idx1 = random.randrange(0, len(db))
        idx2 = random.randrange(0, len(db))
        if idx1 == idx2:
            continue

        f1 = random.choice(db[idx1])
        f2 = random.choice(db[idx2])
        if idx1 > idx2:
            pair = (idx2, idx1, f2, f1)
        else:
            pair = (idx1, idx2, f1, f2)
        mismatched_set.add(pair)

    for pair in mismatched_set:
        idx1, idx2, f1, f2 = pair
        #print("%d %d %s %s" % (idx1, idx2, f1, f2))
        print("%s %s" % (f1, f2))

    return mismatched_set


def do_select(db, matched, mismatched):
    matched_set = do_select_matched(db, matched)
    mismatched_set = do_select_mismatched(db, mismatched)
    # union two sets
    pair_set = matched_set | mismatched_set

    print(pair_set)

    return pair_set

def face_list_write(face_pair_set):

    f_l = open("face_list.txt", "w")
    f_l_gt = open("face_list_gt.txt", "w")

    for pair in face_pair_set:
        idx1, idx2, f1, f2 = pair
        f_l.write("{} {}\n".format(f1, f2))
        if idx1 == idx2:
            label = 1
        else:
            label = 0
        f_l_gt.write("{} {} {}\n".format(label, f1, f2))

    f_l.close()
    f_l_gt.close()


if __name__ == '__main__':

    img_folder = sys.argv[1]
    intra_pair_cnt = int(sys.argv[2])
    inter_pair_cnt = int(sys.argv[3])
    db = image_db_build(img_folder=img_folder, select_matched_count=intra_pair_cnt, select_mismatched_count=inter_pair_cnt)
    pair_set = do_select(db, intra_pair_cnt, inter_pair_cnt)
    face_list_write(pair_set)
