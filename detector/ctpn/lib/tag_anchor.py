import numpy as np
import math


def cal_IoU(cy1, h1, cy2, h2):
    y_top1, y_bottom1 = cal_y(cy1, h1)
    y_top2, y_bottom2 = cal_y(cy2, h2)
    offset = min(y_top1, y_top2)
    y_top1 = y_top1 - offset
    y_top2 = y_top2 - offset
    y_bottom1 = y_bottom1 - offset
    y_bottom2 = y_bottom2 - offset
    line = np.zeros(max(y_bottom1, y_bottom2) + 1)
    for i in range(y_top1, y_bottom1 + 1):
        line[i] += 1
    for j in range(y_top2, y_bottom2 + 1):
        line[j] += 1
    union = np.count_nonzero(line, 0)
    intersection = line[line == 2].size
    return float(intersection)/float(union)


def cal_y(cy, h):
    y_top = int(cy - (float(h) - 1) / 2.0)
    y_bottom = int(cy + (float(h) - 1) / 2.0)
    return y_top, y_bottom


def valid_anchor(cy, h, height):
    top, bottom = cal_y(cy, h)
    if top < 0:
        return False
    if bottom > (height * 16 - 1):
        return False
    return True


def tag_anchor(gt_anchor, cnn_output, gt_box):
    anchor_height = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]  # from 11 to 273, divide 0.7 each time
    # whole image h and w
    height = cnn_output.shape[2]
    width = cnn_output.shape[3]
    positive = []
    negative = []
    vertical_reg = []
    side_refinement_reg = []
    x_left_side = min(gt_box[0], gt_box[6])
    x_right_side = max(gt_box[2], gt_box[4])
    left_side = False
    right_side = False
    for a in gt_anchor:

        if a[0] >= int(width - 1):
            continue

        if x_left_side in range(a[0] * 16, (a[0] + 1) * 16):
            left_side = True
        else:
            left_side = False

        if x_right_side in range(a[0] * 16, (a[0] + 1) * 16):
            right_side = True
        else:
            right_side = False

        iou = np.zeros((height, len(anchor_height)))
        temp_positive = []
        for i in range(iou.shape[0]):
            for j in range(iou.shape[1]):
                if not valid_anchor((float(i) * 16.0 + 7.5), anchor_height[j], height):
                    continue
                iou[i][j] = cal_IoU((float(i) * 16.0 + 7.5), anchor_height[j], a[1], a[2])

                if iou[i][j] > 0.7:
                    temp_positive.append((a[0], i, j, iou[i][j]))
                    if left_side:
                        o = (float(x_left_side) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                        side_refinement_reg.append((a[0], i, j, o))
                    if right_side:
                        o = (float(x_right_side) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                        side_refinement_reg.append((a[0], i, j, o))

                if iou[i][j] < 0.5:
                    negative.append((a[0], i, j, iou[i][j]))

                if iou[i][j] > 0.5:
                    vc = (a[1] - (float(i) * 16.0 + 7.5)) / float(anchor_height[j])
                    vh = math.log10(float(a[2]) / float(anchor_height[j]))
                    vertical_reg.append((a[0], i, j, vc, vh, iou[i][j]))

        if len(temp_positive) == 0:
            max_position = np.where(iou == np.max(iou))
            temp_positive.append((a[0], max_position[0][0], max_position[1][0], np.max(iou)))

            if left_side:
                o = (float(x_left_side) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                side_refinement_reg.append((a[0], max_position[0][0], max_position[1][0], o))
            if right_side:
                o = (float(x_right_side) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                side_refinement_reg.append((a[0], max_position[0][0], max_position[1][0], o))

            if np.max(iou) <= 0.5:
                vc = (a[1] - (float(max_position[0][0]) * 16.0 + 7.5)) / float(anchor_height[max_position[1][0]])
                vh = math.log10(float(a[2]) / float(anchor_height[max_position[1][0]]))
                vertical_reg.append((a[0], max_position[0][0], max_position[1][0], vc, vh, np.max(iou)))
        positive += temp_positive
    return positive, negative, vertical_reg, side_refinement_reg
