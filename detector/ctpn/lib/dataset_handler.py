
import sys
sys.path.append("../..")
import os
import codecs
import cv2
import draw_image
import lmdb

import Net as Net
from torch.utils.data import Dataset


def read_gt_file(path, have_BOM=False):
    result = []
    if have_BOM:
        fp = codecs.open(path, 'r', 'utf-8-sig')
    else:
        fp = open(path, 'r')
    for line in fp.readlines():
        pt = line.split(',')
        if have_BOM:
            box = [int(round(float(pt[i]))) for i in range(8)]
        else:
            box = [int(round(float(pt[i]))) for i in range(8)]
        result.append(box)
    fp.close()
    return result


def create_dataset_icdar2015(img_root, gt_root, output_path):
    im_list = os.listdir(img_root)
    im_path_list = []
    gt_list = []
    for im in im_list:
        name, _ = os.path.splitext(im)
        gt_name = 'gt_' + name + '.txt'
        gt_path = os.path.join(gt_root, gt_name)
        if not os.path.exists(gt_path):
            print('Ground truth file of image {0} not exists.'.format(im))
        im_path_list.append(os.path.join(img_root, im))
        gt_list.append(gt_path)
    assert len(im_path_list) == len(gt_list)
    create_dataset(output_path, im_path_list, gt_list)


def scale_img(img, gt, shortest_side=600):
    height = img.shape[0]
    width = img.shape[1]
    scale = float(shortest_side)/float(min(height, width))
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    if img.shape[0] < img.shape[1] and img.shape[0] != 600:
        img = cv2.resize(img, (600, img.shape[1]))
    elif img.shape[0] > img.shape[1] and img.shape[1] != 600:
        img = cv2.resize(img, (img.shape[0], 600))
    elif img.shape[0] != 600:
        img = cv2.resize(img, (600, 600))
    h_scale = float(img.shape[0])/float(height)
    w_scale = float(img.shape[1])/float(width)
    scale_gt = []
    for box in gt:
        scale_box = []
        for i in range(len(box)):
            if i % 2 == 0:
                scale_box.append(int(int(box[i]) * w_scale))
            else:
                scale_box.append(int(int(box[i]) * h_scale))
        scale_gt.append(scale_box)
    return img, scale_gt


def scale_img_only(img, shortest_side=600):
    height = img.shape[0]
    width = img.shape[1]
    scale = float(shortest_side)/float(min(height, width))
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    if img.shape[0] < img.shape[1] and img.shape[0] != 600:
        img = cv2.resize(img, (600, img.shape[1]))
    elif img.shape[0] > img.shape[1] and img.shape[1] != 600:
        img = cv2.resize(img, (img.shape[0], 600))
    elif img.shape[0] != 600:
        img = cv2.resize(img, (600, 600))

    return img


def check_img(img):
    if img is None:
        return False
    height, width = img.shape[0], img.shape[1]
    if height * width == 0:
        return False
    return True


def write_cache(env, data):
    with env.begin(write=True) as e:
        for i, l in data.iteritems():
            e.put(i, l)


def box_list2str(l):
    result = []
    for box in l:
        if not len(box) % 8 == 0:
            return '', False
        result.append(','.join(box))
    return '|'.join(result), True


def create_dataset(output_path, img_list, gt_list):
    assert len(img_list) == len(gt_list)
    net = Net.VGG_16()
    num = len(img_list)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    env = lmdb.open(output_path, map_size=1099511627776)
    cache = {}
    counter = 1
    for i in range(num):
        img_path = img_list[i]
        gt = gt_list[i]
        if not os.path.exists(img_path):
            print("{0} is not exist.".format(img_path))
            continue

        if len(gt) == 0:
            print("Ground truth of {0} is not exist.".format(img_path))
            continue

        img = cv2.imread(img_path)
        if not check_img(img):
            print('Image {0} is not valid.'.format(img_path))
            continue

        img, gt = scale_img(img, gt)
        gt_str = box_list2str(gt)
        if not gt_str[1]:
            print("Ground truth of {0} is not valid.".format(img_path))
            continue

        img_key = 'image-%09d' % counter
        gt_key = 'gt-%09d' % counter
        cache[img_key] = draw_image.np_img2base64(img, img_path)
        cache[gt_key] = gt_str[0]
        counter += 1
        if counter % 100 == 0:
            write_cache(env, cache)
            cache.clear()
            print('Written {0}/{1}'.format(counter, num))
    cache['num'] = str(counter - 1)
    write_cache(env, cache)
    print('Create dataset with {0} image.'.format(counter - 1))


class LmdbDataset(Dataset):
    def __init__(self, root, transformer=None):
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print("Cannot create lmdb from root {0}.".format(root))
        with self.env.begin(write=False) as e:
            self.data_num = int(e.get('num'))
        self.transformer = transformer

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        assert index <= len(self), 'Index out of range.'
        index += 1
        with self.env.begin(write=False) as e:
            img_key = 'image-%09d' % index
            img_base64 = e.get(img_key)
            img = draw_image.base642np_image(img_base64)
            gt_key = 'gt-%09d' % index
            gt = str(e.get(gt_key))
        return img, gt
