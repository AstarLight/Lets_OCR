import torch.nn as nn
import torch.nn.functional as F
import img2col
import numpy as np
import torch


class VGG_16(nn.Module):
    """
    VGG-16 without pooling layer before fc layer
    """
    def __init__(self):
        super(VGG_16, self).__init__()
        self.convolution1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.convolution1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pooling1 = nn.MaxPool2d(2, stride=2)
        self.convolution2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.convolution2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pooling2 = nn.MaxPool2d(2, stride=2)
        self.convolution3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.convolution3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.convolution3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pooling3 = nn.MaxPool2d(2, stride=2)
        self.convolution4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.convolution4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.convolution4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pooling4 = nn.MaxPool2d(2, stride=2)
        self.convolution5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.convolution5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.convolution5_3 = nn.Conv2d(512, 512, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.convolution1_1(x), inplace=True)
        x = F.relu(self.convolution1_2(x), inplace=True)
        x = self.pooling1(x)
        x = F.relu(self.convolution2_1(x), inplace=True)
        x = F.relu(self.convolution2_2(x), inplace=True)
        x = self.pooling2(x)
        x = F.relu(self.convolution3_1(x), inplace=True)
        x = F.relu(self.convolution3_2(x), inplace=True)
        x = F.relu(self.convolution3_3(x), inplace=True)
        x = self.pooling3(x)
        x = F.relu(self.convolution4_1(x), inplace=True)
        x = F.relu(self.convolution4_2(x), inplace=True)
        x = F.relu(self.convolution4_3(x), inplace=True)
        x = self.pooling4(x)
        x = F.relu(self.convolution5_1(x), inplace=True)
        x = F.relu(self.convolution5_2(x), inplace=True)
        x = F.relu(self.convolution5_3(x), inplace=True)
        return x


class BLSTM(nn.Module):
    def __init__(self, channel, hidden_unit, bidirectional=True):
        """
        :param channel: lstm input channel num
        :param hidden_unit: lstm hidden unit
        :param bidirectional:
        """
        super(BLSTM, self).__init__()
        self.lstm = nn.LSTM(channel, hidden_unit, bidirectional=bidirectional)

    def forward(self, x):
        """
        WARNING: The batch size of x must be 1.
        """
        x = x.transpose(1, 3)
        recurrent, _ = self.lstm(x[0])
        recurrent = recurrent[np.newaxis, :, :, :]
        recurrent = recurrent.transpose(1, 3)
        return recurrent


class CTPN(nn.Module):
    def __init__(self):
        super(CTPN, self).__init__()
        self.cnn = nn.Sequential()
        self.cnn.add_module('VGG_16', VGG_16())
        self.rnn = nn.Sequential()
        self.rnn.add_module('im2col', img2col.Im2col((3, 3), (1, 1), (1, 1)))
        self.rnn.add_module('blstm', BLSTM(3 * 3 * 512, 128))
        self.FC = nn.Conv2d(256, 512, 1)
        self.vertical_coordinate = nn.Conv2d(512, 2 * 10, 1)
        self.score = nn.Conv2d(512, 2 * 10, 1)
        self.side_refinement = nn.Conv2d(512, 10, 1)

    def forward(self, x, val=False):
        x = self.cnn(x)
        x = self.rnn(x)
        x = self.FC(x)
        x = F.relu(x, inplace=True)
        vertical_pred = self.vertical_coordinate(x)
        score = self.score(x)
        if val:
            score = score.reshape((score.shape[0], 10, 2, score.shape[2], score.shape[3]))
            score = score.squeeze(0)
            score = score.transpose(1, 2)
            score = score.transpose(2, 3)
            score = score.reshape((-1, 2))
            #score = F.softmax(score, dim=1)
            score = score.reshape((10, vertical_pred.shape[2], -1, 2))
            vertical_pred = vertical_pred.reshape((vertical_pred.shape[0], 10, 2, vertical_pred.shape[2],
                                                   vertical_pred.shape[3]))
            vertical_pred = vertical_pred.squeeze(0)
        side_refinement = self.side_refinement(x)
        return vertical_pred, score, side_refinement
