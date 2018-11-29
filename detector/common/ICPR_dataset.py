import os, shutil
import random

ORIGIN_GT_PATH = '/home/ljs/data_ready/ali_icpr/gt_1000'
ORIGIN_IM_PATH = '/home/ljs/data_ready/ali_icpr/image_1000'
TEST_GT_PATH = '/home/ljs/data_ready/ali_icpr/test_gt'
TEST_IM_PATH = '/home/ljs/data_ready/ali_icpr/test_im'

total_lsit = os.listdir(ORIGIN_IM_PATH)
test_list = random.sample(total_lsit, int(len(total_lsit)*0.3))

for name in test_list:
    if not os.path.exists(TEST_GT_PATH):
        os.mkdir(TEST_GT_PATH)
    if not os.path.exists(TEST_IM_PATH):
        os.mkdir(TEST_IM_PATH)
    origin_im_path = os.path.join(ORIGIN_IM_PATH, name)
    basename, _ = os.path.splitext(name)
    original_gt_path = os.path.join(ORIGIN_GT_PATH, 'gt_'+basename+ '.txt')
    shutil.move(original_gt_path, TEST_GT_PATH)
    shutil.move(origin_im_path, TEST_IM_PATH)
