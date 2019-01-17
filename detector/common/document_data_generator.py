#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from argparse import RawTextHelpFormatter
import shutil
from PIL import Image, ImageFilter
from PIL import ImageFont
from PIL import ImageDraw
import random
from copy import deepcopy
import cv2
import math
import numpy as np

train_images_dir = None


def get_box_img(box, angle, cx, cy):
    M = cv2.getRotationMatrix2D(center=(cx, cy), angle=angle, scale=1)
    tl = np.array([box[0], box[1], 1], dtype=np.int32).reshape((-1,1))
    ntl = np.dot(M, tl)
    x1 = int(ntl[0])
    y1 = int(ntl[1])
    tr = np.array([box[2], box[3], 1], dtype=np.int32).reshape((-1,1))
    ntr = np.dot(M, tr)
    x2 = int(ntr[0])
    y2 = int(ntr[1])
    br = np.array([box[4], box[5], 1], dtype=np.int32).reshape((-1,1))
    nbr = np.dot(M, br)
    x3 = int(nbr[0])
    y3 = int(nbr[1])
    bl = np.array([box[6], box[7], 1], dtype=np.int32).reshape((-1,1))
    nbl = np.dot(M, bl)
    x4 = int(nbl[0])
    y4 = int(nbl[1])
    return [x1,y1,x2,y2,x3,y3,x4,y4]


def save_image_label(image, i, label, rotate, underline=False, font=None, blur=False):
    image_name = "long_%s_r%d_%d.png" % (font, rotate, i)
    if underline:
        image_name = "underline_" + image_name
    if blur:
        image_name = "blur_" + image_name
    image_path = os.path.join(train_images_dir, image_name)
    image.save(image_path)
    label_name = os.path.splitext(image_name)[0] + '.txt'
    label_path = os.path.join(train_images_dir, label_name)
    with open(label_path, "w+") as f:
        for line in label:
            loc = "%d,%d,%d,%d,%d,%d,%d,%d,%s" % (line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8])
            f.write(loc)
            #f.write('\n')
    visual_name = "visual_" + image_name
    visual_path = os.path.join(train_images_dir, visual_name)
    draw_labels(image, visual_path, label)


def len_of_sentence(sentence):
    count = 0
    digits_letters = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.,'
    for c in sentence:
        if c in digits_letters:
            count += 1
    return len(sentence) - int(count / 2)


def draw_labels(img, name, labels=[]):
    img = np.array(img)
    for label in labels:
        draw_ploy_4pt(img, label)
    cv2.imwrite(name, img)


def draw_ploy_4pt(img, pt, color=(0, 255, 0), thickness=1):
    pts = np.array([[pt[0], pt[1]], [pt[2], pt[3]], [pt[4], pt[5]], [pt[6], pt[7]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    return cv2.polylines(img, [pts], True, color, thickness)


def read_sentence_dict(dict_path):
    sentences = []
    with open(dict_path, "r") as f:
        sentences = f.readlines()
    return sentences


def randomX(w, image_width):
    margin = 50
    image_width -= margin
    maxX = image_width - w
    newX = random.randint(margin, maxX)
    return newX


class DocumentGenerator(object):
    def __init__(self, width, height, underline=False):
        self.width = width
        self.height = height
        self.underline = underline
        self.char_size = 15
        self.bank_line = 1
        self.bank_line_width = 10
        self.offset = 2
        self.y_margin = 200

    def rotate(self):
        pass

    def draw_underline(self):
        pass

    def build_background(self):
        pass

    def add_noise(self):
        pass

    def put_sentence(self):
        pass

    def build_basic_document(self, font_path, sentence_list=[], rotate=0, underline=False, blur=False):
        # 黑色背景
        new_image_list = []
        sentence_loc = []  # [[x,y,w,h,'ssss'], [x,y,w,h, "aaaa"]]
        local_stentence_loc = []
        img = Image.new("RGB", (self.width, self.height), "white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, self.char_size)
        font_name = os.path.basename(font_path)
        font_basename = os.path.splitext(font_name)[0]
        random.shuffle(sentence_list)
        sentence_num_in_one_image = (self.height - self.y_margin*2) / (self.bank_line_width + self.char_size) - self.bank_line
        print("sentence_num_in_one_image = %d" % sentence_num_in_one_image)
        for i, sentence in enumerate(sentence_list):
            a = int((i+1) % sentence_num_in_one_image)
            #print(a)
            sentence.strip('\n')
            #print(sentence)

            y = self.y_margin + (self.char_size + self.bank_line_width) * int(i % sentence_num_in_one_image)
            w = self.char_size * len_of_sentence(sentence)
            h = self.char_size + 2 * self.offset
            x = randomX(w, self.width)
            x1 = x - self.offset * 2
            y1 = y - self.offset
            x2 = x + w + self.offset
            y2 = y - self.offset
            x3 = x + w + self.offset 
            y3 = y + h
            x4 = x - self.offset * 2
            y4 = y + h

            draw.text((x, y), sentence, (0, 0, 0), font=font)
            if underline:
                draw.line(((x,y+h+3),(x+w,y+h+3)), fill=(0,0,0))
            ploy_box = get_box_img([x1,y1,x2,y2,x3,y3,x4,y4], rotate, int((self.width)/2), int((self.height)/2))
            local_stentence_loc.append((ploy_box[0], ploy_box[1], ploy_box[2], ploy_box[3], ploy_box[4],
                                        ploy_box[5], ploy_box[6], ploy_box[7], sentence))

            if a == 0:
                if rotate != 0:
                    img = img.rotate(rotate, center=(int((self.width)/2), int((self.height)/2)), fillcolor=(255, 255, 255))
                #new_image_list.append(deepcopy(img))

                if blur:
                    img = img.filter(ImageFilter.SMOOTH)

                save_image_label(img, i, local_stentence_loc, rotate, font=font_basename, underline=underline, blur=blur)
                img = Image.new("RGB", (self.width, self.height), "white")
                draw = ImageDraw.Draw(img)
                #loc_save = deepcopy(local_stentence_loc)
                #sentence_loc.append(loc_save)
                #print(loc_save)
                local_stentence_loc = []

        #print(sentence_loc)
        return new_image_list, sentence_loc


def args_parse():
    #解析输入参数
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--out_dir', dest='out_dir',
                        default=None, required=True,
                        help='write a caffe dir')
    parser.add_argument('--font_dir', dest='font_dir',
                        default=None, required=True,
                        help='font dir to to produce images')
    parser.add_argument('--dict_path', dest='dict_path',
                        default=None, required=True,
                        help='sentence library')
    parser.add_argument('--width', dest='width',
                        default=2000, required=True,
                        help='width')
    parser.add_argument('--height', dest='height',
                        default=2000, required=True,
                        help='height')
    parser.add_argument('--rotate', dest='rotate',
                        default=0, required=False,
                        help='max rotate degree 0-45')
    parser.add_argument('--rotate_step', dest='rotate_step',
                        default=1, required=False,
                        help='rotate step for the rotate angle')

    args = vars(parser.parse_args())
    return args

'''
python document_data_generator.py --out_dir ./dataset --font_dir /home/ljs/CPS-OCR/ocr/chinese_fonts  \
    --dict_path ./dict.txt --width 600 --height 1200 --rotate 6 --rotate_step 2
'''
if __name__ == "__main__":
    options = args_parse()
    out_dir = os.path.expanduser(options['out_dir'])
    font_dir = os.path.expanduser(options['font_dir'])
    dict_path = options['dict_path']
    width = int(options['width'])
    height = int(options['height'])
    rotate = int(options['rotate'])
    rotate_step = int(options['rotate_step'])
    train_image_dir_name = "train"
    test_image_dir_name = "test"

    # 将dataset分为train和test两个文件夹分别存储
    train_images_dir = os.path.join(out_dir, train_image_dir_name)
    test_images_dir = os.path.join(out_dir, test_image_dir_name)

    if os.path.isdir(train_images_dir):
        shutil.rmtree(train_images_dir)
    os.makedirs(train_images_dir)

    # 对于每类字体进行小批量测试
    verified_font_paths = []
    ## search for file fonts
    for font_name in os.listdir(font_dir):
        path_font_file = os.path.join(font_dir, font_name)
        verified_font_paths.append(path_font_file)

    if rotate < 0:
        roate = - rotate

    all_rotate_angles = []
    if rotate > 0 and rotate <= 45:
        for i in range(0, rotate + 1, rotate_step):
            all_rotate_angles.append(i)
        for i in range(-rotate, 0, rotate_step):
            all_rotate_angles.append(i)
        # print(all_rotate_angles)

    dg = DocumentGenerator(width, height, underline=False)

    sentence_list = read_sentence_dict(dict_path)
    print("the len of dict is %d" % len(sentence_list))
    print(sentence_list)
    total_images = []
    total_labels = []
    # start document files create
    for i, verified_font_path in enumerate(verified_font_paths):  # 内层循环是字体
        label_list = []
        image_list = []
        if rotate == 0:
            image_list, label_list = dg.build_basic_document(verified_font_path, sentence_list)
            image_list, label_list = dg.build_basic_document(verified_font_path, sentence_list, blur=True)
            total_images += image_list
            total_labels += label_list
            image_list, label_list = dg.build_basic_document(verified_font_path, sentence_list, underline=True)
            image_list, label_list = dg.build_basic_document(verified_font_path, sentence_list, underline=True, blur=True)
            total_images += image_list
            total_labels += label_list
        else:
            for k in all_rotate_angles:
                image_list, label_list = dg.build_basic_document(verified_font_path, sentence_list, rotate=k)
                total_images += image_list
                total_labels += label_list
                image_list, label_list = dg.build_basic_document(verified_font_path, sentence_list, rotate=k, underline=True)
                total_images += image_list
                total_labels += label_list

    print("We have generated %d images and %d labels." % (len(total_images), len(total_labels)))
"""
    #print(total_labels)

    for i, image in enumerate(total_images):
        image_name = "%d.png" % i
        image_path = os.path.join(train_images_dir, image_name)
        image.save(image_path)
        label_name = "%d.txt" % i
        label_path = os.path.join(train_images_dir, label_name)
        with open(label_path, "w+") as f:
            for line in total_labels[i]:
                #print(line)
                loc = "%d,%d,%d,%d,%d,%d,%d,%d,%s" % (line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8])
                f.write(loc)
                f.write('\n')
        visual_name = "visual_" + image_name
        visual_path = os.path.join(train_images_dir, visual_name)
        draw_labels(image, visual_path, total_labels[i])
"""

