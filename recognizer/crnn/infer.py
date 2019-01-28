
import time
import torch
import os
from torch.autograd import Variable
import lib.utils
import lib.dataset
from PIL import Image
import Net.net as Net
import alphabets
import numpy as np
import cv2

crnn_model_path = './model/mixed_second_finetune_acc97p7.pth'
IMG_ROOT = './test_images'
running_mode = 'gpu'
alphabet = alphabets.alphabet
nclass = len(alphabet) + 1

# Testing images are scaled to have height 32. Widths are
# proportionally scaled with heights, but at least 100 pixels
def scale_img_para(img, min_width=100, fixed_height=32):
    height = img.size[1]
    width = img.size[0]
    scale = float(fixed_height)/height
    w = int(width * scale)
    if w < min_width:
        w = min_width

    return w, fixed_height


def crnn_recognition(cropped_image, model):
    converter = lib.utils.strLabelConverter(alphabet)

    image = cropped_image.convert('L')

    ##
    #w = int(image.size[0] / (280 * 1.0 / 160))
    w, h = scale_img_para(image)
    transformer = lib.dataset.resizeNormalize((w, h))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('results: {0}'.format(sim_pred))


if __name__ == '__main__':

    # crnn network
    model = Net.CRNN(nclass)
    if running_mode == 'gpu' and torch.cuda.is_available():
        model = model.cuda()
        model.load_state_dict(torch.load(crnn_model_path))
    else:
        model.load_state_dict(torch.load(crnn_model_path, map_location='cpu'))

    print('loading pretrained model from {0}'.format(crnn_model_path))

    files = os.listdir(IMG_ROOT)
    for file in files:
        started = time.time()
        full_path = os.path.join(IMG_ROOT, file)
        print("ocr image is %s" % full_path)
        image = Image.open(full_path)

        crnn_recognition(image, model)
        finished = time.time()
        print('elapsed time: {0}'.format(finished - started))