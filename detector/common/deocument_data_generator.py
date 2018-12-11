#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from argparse import RawTextHelpFormatter
import shutil
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import random


def read_sentence_dict(dict_path):
    sentences = []
    with open(dict_path, "r") as f:
        sentences = f.readlines()
    return sentences


class DocumentGenerator(object):
    def __init__(self, width, height, underline=False):
        self.width = width
        self.height = height
        self.underline = underline
        self.char_size = 30
        self.bank_line = 5
        self.bank_line_width = 15

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

    def build_basic_document(self, font_path, sentence_list=[], rotate=0):
        # 黑色背景
        new_image_list = []
        sentence_loc = []  # [[x,y,w,h,'ssss'], [x,y,w,h, "aaaa"]]
        local_stentence_loc = []
        img = Image.new("RGB", (self.width, self.height), "white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, self.char_size)
        random.shuffle(sentence_list)
        sentence_num_in_one_image = self.height / (self.bank_line_width + self.char_size) - self.bank_line

        for i, sentence in enumerate(sentence_list):
            sentence.strip('\r\n')
            print(sentence)
            if i % sentence_num_in_one_image == 0:
                new_image_list.append(img.copy())
                img = Image.new("RGB", (self.width, self.height), "white")
                draw = ImageDraw.Draw(img)
                sentence_loc.append(local_stentence_loc)
                local_stentence_loc = []
            x = 50
            y = 50 + self.char_size*(i % sentence_num_in_one_image + 1)+self.bank_line_width
            w = self.char_size * len(sentence)
            h = self.char_size
            draw.text((x, y), sentence, (0, 0, 0), font=font)
            local_stentence_loc.append((x,y,w,h,sentence))
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
    parser.add_argument('--test_ratio', dest='test_ratio',
                        default=0.2, required=False,
                        help='test dataset size')
    parser.add_argument('--width', dest='width',
                        default=1400, required=True,
                        help='width')
    parser.add_argument('--height', dest='height',
                        default=2000, required=True,
                        help='height')
    parser.add_argument('--no_crop', dest='no_crop',
                        default=True, required=False,
                        help='', action='store_true')
    parser.add_argument('--margin', dest='margin',
                        default=0, required=False,
                        help='', )
    parser.add_argument('--rotate', dest='rotate',
                        default=0, required=False,
                        help='max rotate degree 0-45')
    parser.add_argument('--rotate_step', dest='rotate_step',
                        default=0, required=False,
                        help='rotate step for the rotate angle')
    parser.add_argument('--need_aug', dest='need_aug',
                        default=False, required=False,
                        help='need data augmentation', action='store_true')
    args = vars(parser.parse_args())
    return args

'''
python gen_printed_char.py --out_dir ./dataset --font_dir /home/ljs/CPS-OCR/ocr/chinese_fonts  --width 1400 --height 2000 
'''
if __name__ == "__main__":
    options = args_parse()
    out_dir = os.path.expanduser(options['out_dir'])
    font_dir = os.path.expanduser(options['font_dir'])
    test_ratio = float(options['test_ratio'])
    width = int(options['width'])
    height = int(options['height'])
    need_crop = not options['no_crop']
    margin = int(options['margin'])
    rotate = int(options['rotate'])
    need_aug = options['need_aug']
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

    sentence_list = read_sentence_dict('./dict.txt')
    print(sentence_list)
    total_images = []
    total_labels = []
    # start document files create
    for i, verified_font_path in enumerate(verified_font_paths):  # 内层循环是字体
        label_list = []
        image_list = []
        if rotate == 0:
            image_list, label_list = dg.build_basic_document(verified_font_path, sentence_list)
            total_images += image_list
            total_labels += label_list
        else:
            for k in all_rotate_angles:
                image_list, label_list = dg.build_basic_document(verified_font_path, sentence_list, rotate=k)
                total_images += image_list
                total_labels += label_list

    print("We have generated %d images and %d labels." % (len(total_images), len(total_labels)))

    for i, image in enumerate(total_images):
        image_name = "%d.png" % i
        image_path = os.path.join(train_images_dir, image_name)
        image.save(image_path)
        label_name = "%d.txt" % i
        label_path = os.path.join(train_images_dir, label_name)
        with open(label_path, "w+") as f:
            for line in total_labels[i]:
                loc = "%d,%d,%d,%d,%s" % (line[0], line[1]. line[2]. line[3], line[4])
                f.write(loc)
                f.write('\n')

