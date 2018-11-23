import cv2
import numpy as np
from ..common import draw_img
import base64
import os
import copy
import lib.dataset_handler
import torch
from Net import net as Net
import torchvision.models
anchor_height = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]


if __name__ == '__main__':
    net = Net.CTPN()
    net.load_state_dict(torch.load('./model/ctpn-9-end.model'))
    print(net)
    net.eval()
    im = cv2.imread('../dataset/OCR_dataset/ctpn/test_im/img_0059.jpg')
    img = copy.deepcopy(im)
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]
    img = torch.Tensor(img)
    v, score, side = net(img, val=True)
    result = []
    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            for k in range(score.shape[2]):
                if score[i, j, k, 1] > 0.6:
                    result.append((j, k, i, float(score[i, j, k, 1].detach().numpy())))
    # print(result)
    for box in result:
        im = draw_img.draw_box_h_and_c(im, box[1], box[0] * 16 + 7.5, anchor_height[box[2]])
    gt = lib.dataset_handler.read_gt_file('../dataset/OCR_dataset/ctpn/test_gt/gt_img_0059.txt')
    for gt_box in gt:
        im = draw_img.draw_box_4pt(im, gt_box, (255, 0, 0))

    cv2.imwrite("./test_result/test.jpg", im)