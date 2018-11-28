#coding=utf-8
import cv2
import numpy as np
import os
import copy
import lib.dataset_handler
import torch
import Net.net as Net
import lib.utils
import lib.draw_image
import lib.nms
import math
import random
import shutil
import sys
np.set_printoptions(threshold='nan')
anchor_height = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]

IMG_ROOT = "/home/ljs/ctpn_torch/dataset/OCR_dataset/ctpn/test_im"
GT_ROOT = "/home/ljs/ctpn_torch/dataset/OCR_dataset/ctpn/test_gt"
TEST_RESULT = './test_result'


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::2]=threshold(boxes[:, 0::2], 0, im_shape[1]-1)
    boxes[:, 1::2]=threshold(boxes[:, 1::2], 0, im_shape[0]-1)
    return boxes


def fit_y(X, Y, x1, x2):
    len(X) != 0
    # if X only include one point, the function will get line y=Y[0]
    if np.sum(X == X[0]) == len(X):
        return Y[0], Y[0]
    p = np.poly1d(np.polyfit(X, Y, 1))
    return p(x1), p(x2)


def get_text_lines(text_proposals, im_size, scores=0):
    """
    text_proposals:boxes

    """
    #tp_groups = neighbour_connector(text_proposals, im_size)  # 首先还是建图，获取到文本行由哪几个小框构成
    #print(tp_groups)
    text_lines = np.zeros((len(text_proposals), 8), np.float32)

    for index, tp_indices in enumerate(text_proposals):
        text_line_boxes = np.array(tp_indices)  # 每个文本行的全部小框
        #print(text_line_boxes)
        #print(type(text_line_boxes))
        #print(text_line_boxes.shape)
        X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2  # 求每一个小框的中心x，y坐标
        Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2
        #print(X)
        #print(Y)

        z1 = np.polyfit(X, Y, 1)  # 多项式拟合，根据之前求的中心店拟合一条直线（最小二乘）

        x0 = np.min(text_line_boxes[:, 0])  # 文本行x坐标最小值
        x1 = np.max(text_line_boxes[:, 2])  # 文本行x坐标最大值

        offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5  # 小框宽度的一半

        # 以全部小框的左上角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
        lt_y, rt_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
        # 以全部小框的左下角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
        lb_y, rb_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

        #score = scores[list(tp_indices)].sum() / float(len(tp_indices))  # 求全部小框得分的均值作为文本行的均值

        text_lines[index, 0] = x0
        text_lines[index, 1] = min(lt_y, rt_y)  # 文本行上端 线段 的y坐标的小值
        text_lines[index, 2] = x1
        text_lines[index, 3] = max(lb_y, rb_y)  # 文本行下端 线段 的y坐标的大值
        text_lines[index, 4] = scores  # 文本行得分
        text_lines[index, 5] = z1[0]  # 根据中心点拟合的直线的k，b
        text_lines[index, 6] = z1[1]
        height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))  # 小框平均高度
        text_lines[index, 7] = height + 2.5

    text_recs = np.zeros((len(text_lines), 9), np.float32)
    index = 0
    for line in text_lines:
        b1 = line[6] - line[7] / 2  # 根据高度和文本行中心线，求取文本行上下两条线的b值
        b2 = line[6] + line[7] / 2
        x1 = line[0]
        y1 = line[5] * line[0] + b1  # 左上
        x2 = line[2]
        y2 = line[5] * line[2] + b1  # 右上
        x3 = line[0]
        y3 = line[5] * line[0] + b2  # 左下
        x4 = line[2]
        y4 = line[5] * line[2] + b2  # 右下
        disX = x2 - x1
        disY = y2 - y1
        width = np.sqrt(disX * disX + disY * disY)  # 文本行宽度

        fTmp0 = y3 - y1  # 文本行高度
        fTmp1 = fTmp0 * disY / width
        x = np.fabs(fTmp1 * disX / width)  # 做补偿
        y = np.fabs(fTmp1 * disY / width)
        if line[5] < 0:
            x1 -= x
            y1 += y
            x4 += x
            y4 -= y
        else:
            x2 += x
            y2 += y
            x3 -= x
            y3 -= y
        # clock-wise order
        text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x4
        text_recs[index, 5] = y4
        text_recs[index, 6] = x3
        text_recs[index, 7] = y3
        text_recs[index, 8] = line[4]
        index = index + 1

    text_recs = clip_boxes(text_recs, im_size)

    return text_recs


def meet_v_iou(y1, y2, h1, h2):
    def overlaps_v(y1, y2, h1, h2):
        return max(0, y2-y1+1)/min(h1, h2)

    def size_similarity(h1, h2):
        return min(h1, h2)/max(h1, h2)

    return overlaps_v(y1, y2, h1, h2) >= 0.6 and \
           size_similarity(h1, h2) >= 0.6


def get_successions(index, im_size, text_proposals, boxes_table):
        box=text_proposals[index]
        heights = text_proposals[:, 3] - text_proposals[:, 1] + 1
        results=[]
        # find horizon
        for left in range(int(box[0])+1, min(int(box[0])+30+1, im_size[1])):
            adj_box_indices=boxes_table[left]  # vertical adjacent anchors, find vertical
            for adj_box_index in adj_box_indices:
                if meet_v_iou(adj_box_index, index, text_proposals, heights):
                    #print(text_proposals[adj_box_index])
                    results.append(adj_box_index)
            if len(results)!=0:
                return results
        return results


def neighbour_connector(text_proposals, im_size):
    print(text_proposals)
    boxes_table = [[] for _ in range(im_size[1])]
    for index, box in enumerate(text_proposals):
        boxes_table[int(box[0])].append(index)
    print(boxes_table)
    successions = []
    for index, box in enumerate(text_proposals):
        s = get_successions(index, im_size, text_proposals, boxes_table)
        if len(s) == 0:
            continue
        successions.append(s)
    return successions


def gen_test_images(gt_root, img_root, test_num=10):
    img_list = os.listdir(img_root)
    #random_list = random.sample(img_list, test_num)
    random_list = img_list
    test_pair = []
    for im in random_list:
        name, _ = os.path.splitext(im)
        gt_name = 'gt_' + name + '.txt'
        gt_path = os.path.join(gt_root, gt_name)
        im_path = os.path.join(img_root, im)
        test_pair.append((im_path, gt_path))
    return test_pair


def get_anchor_h(anchor, v):
    vc = v[int(anchor[7]), 0, int(anchor[5]), int(anchor[6])]
    vh = v[int(anchor[7]), 1, int(anchor[5]), int(anchor[6])]
    cya = anchor[5] * 16 + 7.5
    ha = anchor_height[int(anchor[7])]
    cy = vc * ha + cya
    h = math.pow(10, vh) * ha
    return h


def get_text_anchors(v, anchors=[]):
    anchor_graph = np.zeros(len(anchors), np.bool)
    texts = []
    for i, anchor in enumerate(anchors):
        one_text = []
        if not anchor_graph[i]:
            one_text.append(anchor)
            anchor_graph[i] = True
            center_x1 = (anchor[2] + anchor[0])/2
            center_y1 = (anchor[3] + anchor[1]) / 2
            h1 = get_anchor_h(anchor, v)
            for j in range(i+1, len(anchors)):
                if not anchor_graph[j]:
                    center_x2 = (anchors[j][2] + anchors[j][0])/2
                    center_y2 = (anchors[j][3] + anchors[j][1]) / 2
                    h2 = get_anchor_h(anchors[j], v)
                    print("h1 is %s, h2 is %s" % (h1,h2))
                    if abs(center_x1-center_x2) < 50 and \
                            meet_v_iou(max(anchor[1], anchors[j][1]), min(anchor[3], anchors[j][3]), h1, h2):   # less than 50 pixel between each anchor
                        one_text.append(anchors[j])
                        #print(anchors[j])
                        anchor_graph[j] = True
        if len(one_text) != 0:
            texts.append(one_text)
            print(one_text)
    return texts


def get_ssss(v, anchors=[]):
    texts = []
    for i, anchor in enumerate(anchors):
        neighbours = []
        neighbours.append(i)
        center_x1 = (anchor[2] + anchor[0]) / 2
        center_y1 = (anchor[3] + anchor[1]) / 2
        h1 = get_anchor_h(anchor, v)
        # find i's neighbour
        for j in range(i + 1, len(anchors)):
            center_x2 = (anchors[j][2] + anchors[j][0]) / 2
            center_y2 = (anchors[j][3] + anchors[j][1]) / 2
            h2 = get_anchor_h(anchors[j], v)
            print("h1 is %s, h2 is %s" % (h1, h2))
            if abs(center_x1 - center_x2) < 50 and \
                    meet_v_iou(max(anchor[1], anchors[j][1]), min(anchor[3], anchors[j][3]), h1, h2):  # less than 50 pixel between each anchor
                neighbours.append(j)

        # now we get i's neighbours, then we find if i also locate in somewhere
        for k, line in enumerate(texts):
            for index in line:
                if index == i:
                    texts[k] += neighbours
                    texts[k] = list(set(texts[k]))
                    neighbours = []
        if len(neighbours) != 0:
            texts.append(neighbours)

    # ok, we combine again.
    for i, line in enumerate(texts):
        if len(line) == 0:
            continue
        for index in line:
            for j in range(i+1, len(texts)):
                if index in texts[j]:
                    texts[i] += texts[j]
                    texts[i] = list(set(texts[i]))
                    texts[j] = []

    result = []
    print(texts)
    for text in texts:
        if len(text) < 2:
            continue
        local = []
        for j in text:
            local.append(anchors[j])
        result.append(local)
    return result


def infer_one(im_name, net):
    im = cv2.imread(im_name)
    im = lib.dataset_handler.scale_img_only(im)
    img = copy.deepcopy(im)
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]
    img = torch.Tensor(img)
    v, score, side = net(img, val=True)
    result = []
    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            for k in range(score.shape[2]):
                if score[i, j, k, 1] > 0.7:
                    result.append((j, k, i, float(score[i, j, k, 1].detach().numpy())))

    for_nms = []
    for box in result:
        pt = lib.utils.trans_to_2pt(box[1], box[0] * 16 + 7.5, anchor_height[box[2]])
        for_nms.append([pt[0], pt[1], pt[2], pt[3], box[3], box[0], box[1], box[2]])
    for_nms = np.array(for_nms, dtype=np.float32)
    nms_result = lib.nms.cpu_nms(for_nms, 0.3)

    out_nms = []
    for i in nms_result:
        out_nms.append(for_nms[i, 0:8])

    #print(out_nms)
    print(type(out_nms))
    connect = get_ssss(v, out_nms)
    print("size of texts %s" % len(connect))
    texts = get_text_lines(connect, im.shape)
    #print(texts)
    for box in texts:
        box = np.array(box)
        print(box)
        lib.draw_image.draw_ploy_4pt(im, box[0:8])

    _, basename = os.path.split(im_name)
    cv2.imwrite('./infer_'+basename, im)

    for i in nms_result:
        vc = v[int(for_nms[i, 7]), 0, int(for_nms[i, 5]), int(for_nms[i, 6])]
        vh = v[int(for_nms[i, 7]), 1, int(for_nms[i, 5]), int(for_nms[i, 6])]
        cya = for_nms[i, 5] * 16 + 7.5
        ha = anchor_height[int(for_nms[i, 7])]
        cy = vc * ha + cya
        h = math.pow(10, vh) * ha
        lib.draw_image.draw_box_2pt(im, for_nms[i, 0:4])
    _, basename = os.path.split(im_name)
    cv2.imwrite('./infer_anchor_'+basename, im)


def random_test(net):
    test_pair = gen_test_images(GT_ROOT, IMG_ROOT, 50)
    print(test_pair)
    if os.path.exists(TEST_RESULT):
        shutil.rmtree(TEST_RESULT)

    os.mkdir(TEST_RESULT)

    for t in test_pair:
        im = cv2.imread(t[0])
        gt = lib.dataset_handler.read_gt_file(t[1])
        im, gt = lib.dataset_handler.scale_img(im, gt)
        img = copy.deepcopy(im)
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, :, :, :]
        img = torch.Tensor(img)
        v, score, side = net(img, val=True)
        result = []
        for i in range(score.shape[0]):
            for j in range(score.shape[1]):
                for k in range(score.shape[2]):
                    if score[i, j, k, 1] > 0.7:
                        result.append((j, k, i, float(score[i, j, k, 1].detach().numpy())))

        for_nms = []
        for box in result:
            pt = lib.utils.trans_to_2pt(box[1], box[0] * 16 + 7.5, anchor_height[box[2]])
            for_nms.append([pt[0], pt[1], pt[2], pt[3], box[3], box[0], box[1], box[2]])
        for_nms = np.array(for_nms, dtype=np.float32)
        nms_result = lib.nms.cpu_nms(for_nms, 0.3)



        out_nms = []
        for i in nms_result:
            out_nms.append(for_nms[i, 0:8])

        # print(out_nms)
        #print(type(out_nms))
        connect = get_ssss(v, out_nms)
        #print("size of texts %s" % len(connect))
        texts = get_text_lines(connect, im.shape)
        # print(texts)

        for i in nms_result:
            vc = v[int(for_nms[i, 7]), 0, int(for_nms[i, 5]), int(for_nms[i, 6])]
            vh = v[int(for_nms[i, 7]), 1, int(for_nms[i, 5]), int(for_nms[i, 6])]
            cya = for_nms[i, 5] * 16 + 7.5
            ha = anchor_height[int(for_nms[i, 7])]
            cy = vc * ha + cya
            h = math.pow(10, vh) * ha
            lib.draw_image.draw_box_2pt(im, for_nms[i, 0:4])
            #im = other.draw_box_h_and_c(im, int(for_nms[i, 6]), cy, h)

        for box in texts:
            box = np.array(box)
            print(box)
            lib.draw_image.draw_ploy_4pt(im, box[0:8], thickness=4)
        cv2.imwrite(os.path.join(TEST_RESULT, os.path.basename(t[0])), im)
"""
        for i in nms_result:
            vc = v[int(for_nms[i, 7]), 0, int(for_nms[i, 5]), int(for_nms[i, 6])]
            vh = v[int(for_nms[i, 7]), 1, int(for_nms[i, 5]), int(for_nms[i, 6])]
            cya = for_nms[i, 5] * 16 + 7.5
            ha = anchor_height[int(for_nms[i, 7])]
            cy = vc * ha + cya
            h = math.pow(10, vh) * ha
            lib.draw_image.draw_box_2pt(im, for_nms[i, 0:4])
            #im = other.draw_box_h_and_c(im, int(for_nms[i, 6]), cy, h)

        # print(result)
        #for box in result:
         #   im = other.draw_box_h_and_c(im, box[1], box[0] * 16 + 7.5, anchor_height[box[2]])

        for gt_box in gt:
            im = lib.draw_image.draw_box_4pt(im, gt_box, (255, 0, 0))

        cv2.imwrite(os.path.join(TEST_RESULT, os.path.basename(t[0])), im)
"""

if __name__ == '__main__':
    net = Net.CTPN()
    net.load_state_dict(torch.load('./model/ctpn-29-end.model'))
    print(net)
    net.eval()

    if sys.argv[1] == 'random':
        random_test(net)
    else:
        url = sys.argv[1]
        infer_one(url, net)



