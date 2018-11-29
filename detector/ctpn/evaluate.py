import torch
import cv2
import lib.dataset_handler
import lib.generate_gt_anchor
import lib.tag_anchor
import Net
import numpy as np
import os
import time
import random
import copy


def val(net, criterion, batch_num, using_cuda, logger, img_list):
    random_list = random.sample(img_list, batch_num)
    total_loss = 0
    total_cls_loss = 0
    total_v_reg_loss = 0
    total_o_reg_loss = 0
    start_time = time.time()
    for im in random_list:
        root, file_name = os.path.split(im)
        root, _ = os.path.split(root)
        name, _ = os.path.splitext(file_name)
        gt_name = 'gt_' + name + '.txt'
        gt_path = os.path.join(root, "test_gt", gt_name)
        if not os.path.exists(gt_path):
            print('Ground truth file of image {0} not exists.'.format(gt_path))
            continue

        gt_txt = lib.dataset_handler.read_gt_file(gt_path, have_BOM=True)
        img = cv2.imread(im)
        if img is None:
            batch_num -= 1
            continue

        img, gt_txt = lib.dataset_handler.scale_img(img, gt_txt)
        tensor_img = img[np.newaxis, :, :, :]
        tensor_img = tensor_img.transpose((0, 3, 1, 2))
        if using_cuda:
            tensor_img = torch.FloatTensor(tensor_img).cuda()
        else:
            tensor_img = torch.FloatTensor(tensor_img)

        vertical_pred, score, side_refinement = net(tensor_img)
        del tensor_img
        positive = []
        negative = []
        vertical_reg = []
        side_refinement_reg = []
        visual_img = copy.deepcopy(img)

        try:
            for box in gt_txt:
                gt_anchor, visual_img = lib.generate_gt_anchor.generate_gt_anchor(img, box, draw_img_gt=visual_img)
                positive1, negative1, vertical_reg1, side_refinement_reg1 = lib.tag_anchor.tag_anchor(gt_anchor, score, box)
                positive += positive1
                negative += negative1
                vertical_reg += vertical_reg1
                side_refinement_reg += side_refinement_reg1
        except:
            print("warning: img %s raise error!" % im)
            batch_num -= 1
            continue

        if len(vertical_reg) == 0 or len(positive) == 0 or len(side_refinement_reg) == 0:
            batch_num -= 1
            continue

        loss, cls_loss, v_reg_loss, o_reg_loss = criterion(score, vertical_pred, side_refinement, positive,
                                                           negative, vertical_reg, side_refinement_reg)
        total_loss += float(loss)
        total_cls_loss += float(cls_loss)
        total_v_reg_loss += float(v_reg_loss)
        total_o_reg_loss += float(o_reg_loss)
    end_time = time.time()
    total_time = end_time - start_time
    print('####################  Start evaluate  ####################')
    print('loss: {0}'.format(total_loss / float(batch_num)))
    logger.info('Evaluate loss: {0}'.format(total_loss / float(batch_num)))

    print('classification loss: {0}'.format(total_cls_loss / float(batch_num)))
    logger.info('Evaluate vertical regression loss: {0}'.format(total_v_reg_loss / float(batch_num)))

    print('vertical regression loss: {0}'.format(total_v_reg_loss / float(batch_num)))
    logger.info('Evaluate side-refinement regression loss: {0}'.format(total_o_reg_loss / float(batch_num)))

    print('side-refinement regression loss: {0}'.format(total_o_reg_loss / float(batch_num)))
    logger.info('Evaluate side-refinement regression loss: {0}'.format(total_o_reg_loss / float(batch_num)))

    print('{1} iterations for {0} seconds.'.format(total_time, batch_num))
    print('#####################  Evaluate end  #####################')
    print('\n')
    return total_loss
