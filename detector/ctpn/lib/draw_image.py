import cv2
import numpy as np
import base64
import os
import torch


def draw_box_4pt(img, pt, color=(0, 255, 0), thickness=2):
    if not isinstance(pt[0], int):
        pt = [int(pt[i]) for i in range(8)]
    img = cv2.line(img, (pt[0], pt[1]), (pt[2], pt[3]), color, thickness)
    img = cv2.line(img, (pt[2], pt[3]), (pt[4], pt[5]), color, thickness)
    img = cv2.line(img, (pt[4], pt[5]), (pt[6], pt[7]), color, thickness)
    img = cv2.line(img, (pt[6], pt[7]), (pt[0], pt[1]), color, thickness)
    return img


def draw_ploy_4pt(img, pt, color=(0, 255, 255), thickness=1):
    pts = np.array([[pt[0], pt[1]], [pt[2], pt[3]], [pt[4], pt[5]], [pt[6], pt[7]]], np.int32)
    print(pts)
    pts = pts.reshape((-1, 1, 2))
    return cv2.polylines(img, [pts], True, color, thickness)


def draw_box_2pt(img, pt, color=(0, 255, 0), thickness=1):
    if not isinstance(pt[0], int):
        pt = [int(pt[i]) for i in range(4)]
    img = cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), color, thickness=thickness)
    return img


def draw_box_h_and_c(img, position, cy, h, anchor_width=16, color=(0, 255, 0), thickness=1):
    x_left = position * anchor_width
    x_right = (position + 1) * anchor_width - 1
    y_top = int(cy - (float(h) - 1) / 2.0)
    y_bottom = int(cy + (float(h) - 1) / 2.0)
    pt = [x_left, y_top, x_right, y_bottom]
    return draw_box_2pt(img, pt, color=color, thickness=thickness)


def np_img2base64(np_img, path):
    image = cv2.imencode(os.path.splitext(path)[1], np_img)[1]
    image = np.squeeze(image, 1)
    image_code = base64.b64encode(image)
    return image_code


def base642np_image(base64_str):
    missing_padding = 4 - len(base64_str) % 4
    if missing_padding:
        base64_str += b'=' * missing_padding
    raw_str = base64.b64decode(base64_str)
    np_img = np.fromstring(raw_str, dtype=np.uint8)
    img = cv2.imdecode(np_img, cv2.COLOR_RGB2BGR)
    return img


def cal_line_y(pt1, pt2, x, form):
    if not isinstance(pt1[0], float) or not isinstance(pt2[0], float):
        pt1 = [float(pt1[i]) for i in range(len(pt1))]
        pt2 = [float(pt2[i]) for i in range(len(pt2))]
    if not isinstance(x, float):
        x = float(x)
    if (pt1[0] - pt2[0]) == 0:
        return -1
    return form(((pt1[1] - pt2[1])/(pt1[0] - pt2[0])) * (x - pt1[0]) + pt1[1])


def bi_range(start, end):
    start = int(start)
    end = int(end)
    if start > end:
        return range(end, start)
    else:
        return range(start, end)


def init_weight(net):
    for i in range(len(net.rnn.blstm.lstm.all_weights)):
        for j in range(len(net.rnn.blstm.lstm.all_weights[0])):
            torch.nn.init.normal_(net.rnn.blstm.lstm.all_weights[i][j], std=0.01)

    torch.nn.init.normal_(net.FC.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.FC.bias, val=0)

    torch.nn.init.normal_(net.vertical_coordinate.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.vertical_coordinate.bias, val=0)

    torch.nn.init.normal_(net.score.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.score.bias, val=0)

    torch.nn.init.normal_(net.side_refinement.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.side_refinement.bias, val=0)
