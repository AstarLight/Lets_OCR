
import time
import torch
import os
from torch.autograd import Variable
import lib.utils
import lib.dataset
from PIL import Image
import Net.net as Net
import alphabets

crnn_model_path = 'trained_models/mixed_second_finetune_acc97p7.pth'
IMG_ROOT = './test'
alphabet = alphabets.alphabet
nclass = len(alphabet) + 1


def crnn_recognition(cropped_image, model):
    converter = lib.utils.strLabelConverter(alphabet)

    image = cropped_image.convert('L')

    ##
    w = int(image.size[0] / (280 * 1.0 / 160))
    transformer = lib.dataset.resizeNormalize((w, 32))
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
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from {0}'.format(crnn_model_path))

    model.load_state_dict(torch.load(crnn_model_path))

    files = os.listdir(IMG_ROOT)
    for file in files:
        started = time.time()
        full_path = os.path.join(IMG_ROOT, file)
        print("ocr image is %s" % full_path)
        image = Image.open(full_path)

        crnn_recognition(image, model)
        finished = time.time()
        print('elapsed time: {0}'.format(finished - started))