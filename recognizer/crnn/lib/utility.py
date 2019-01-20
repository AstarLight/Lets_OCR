import cv2
import os

import torch
from torch.autograd import Variable
import random
import dataset


def scale_image(img, height, width, keep_ratio=False):
    if not keep_ratio:
        result = cv2.resize(img, (width, height))
    else:
        fy = float(height)/float(img.shape[0])
        result = cv2.resize(img, (0, 0), fx=fy, fy=fy)
        print(result.shape)
        if result.shape[1] < width:
            fx = float(width)/float(result.shape[1])
            result = cv2.resize(result, (0, 0), fx=fx, fy=1)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    return result


def get_all_file_path(folder, file_ext=None):
    """
    :param folder: folder of all files
    :param file_ext: type of file
    :return: list of files
    """
    result = []
    if file_ext is None:
        for f in os.walk(folder):
            for file in f[2]:
                result.append(os.path.join(f[0], file).replace('\\', '/'))
    else:
        for f in os.walk(folder):
            for file in f[2]:
                if os.path.splitext(file)[1] in file_ext:
                    result.append(os.path.join(f[0], file).replace('\\', '/'))
                else:
                    pass
    return result


# def get_batch_data(file_list, batch_size=64, width=100, height=32):
#     sample = random.sample(file_list, batch_size)
#     im = []
#     text = []
#     for i in sample:
#         im.append(cv2.imread(i))
#         label = os.path.splitext(i.split('/')[-1])[0]
#         label = label.split('_')[1]
#         text.append(label)
#     im = np.array(im)
#     return im, text


def create_dataset_port(imgFolderPath, train_outputPath, test_outputPath, train_ratio=0.9, fileExt=None):
    """
    for synth 90k dataset
    """
    im_list = get_all_file_path(imgFolderPath, file_ext=fileExt)
    random.shuffle(im_list)
    im_num = len(im_list)
    train_num = int(im_num * train_ratio)

    print('Total img number: {0}'.format(im_num))
    print('Train img number: {0}'.format(train_num))
    print('Test img number: {0}'.format(im_num - train_num))

    train_list = im_list[0:(train_num - 1)]
    test_list = im_list[train_num:-1]
    train_label_list = []
    test_label_list = []
    for i in train_list:
        label = os.path.splitext((i.split('/')[-1]))[0]
        label = label.split('_')[1]
        train_label_list.append(label)
    print('Generate train database.')
    dataset.createDataset(train_outputPath, train_list, train_label_list)

    for j in test_list:
        label = os.path.splitext((j.split('/')[-1]))[0]
        label = label.split('_')[1]
        test_label_list.append(label)
    print('Generate test database.')
    dataset.createDataset(test_outputPath, test_list, test_label_list)


def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)