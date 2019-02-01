
import time
import torch
import os
from torch.autograd import Variable
import lib.convert
import lib.dataset
from PIL import Image
import Net.net as Net
import alphabets
import sys
import Config

os.environ['CUDA_VISIBLE_DEVICES'] = "4"

crnn_model_path = './w160_bs64_model/netCRNN_4_32000.pth'
IMG_ROOT = './test_images'
running_mode = 'gpu'
alphabet = alphabets.alphabet
nclass = len(alphabet) + 1


def crnn_recognition(cropped_image, model):
    converter = lib.convert.strLabelConverter(alphabet)

    image = cropped_image.convert('L')

    ### Testing images are scaled to have height 32. Widths are
    # proportionally scaled with heights, but at least 100 pixels
    w = int(image.size[0] / (280 * 1.0 / Config.infer_img_w))
    #scale = image.size[1] * 1.0 / Config.img_height
    #w = int(image.size[0] / scale)

    transformer = lib.dataset.resizeNormalize((w, Config.img_height))
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

    files = sorted(os.listdir(IMG_ROOT))
    for file in files:
        started = time.time()
        full_path = os.path.join(IMG_ROOT, file)
        print("=============================================")
        print("ocr image is %s" % full_path)
        image = Image.open(full_path)

        crnn_recognition(image, model)
        finished = time.time()
        print('elapsed time: {0}'.format(finished - started))

    sys.exit()

