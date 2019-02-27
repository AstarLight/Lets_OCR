
import pretrainedmodels as pm

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
import math

SPEEDUP_SCALE = 512


class DummyLayer(nn.Module):

    def forward(self, input_f):
        return input_f


# f + g = h layer
class HLayer(nn.Module):

    def __init__(self, inputChannels, outputChannels):
        """

        :param inputChannels: channels of g+f
        :param outputChannels:
        """
        super(HLayer, self).__init__()

        self.conv2dOne = nn.Conv2d(inputChannels, outputChannels, kernel_size=1)
        self.bnOne = nn.BatchNorm2d(outputChannels, momentum=0.003)

        self.conv2dTwo = nn.Conv2d(outputChannels, outputChannels, kernel_size=3, padding=1)
        self.bnTwo = nn.BatchNorm2d(outputChannels, momentum=0.003)

    def forward(self, inputPrevG, inputF):
        input = torch.cat([inputPrevG, inputF], dim=1)
        output = self.conv2dOne(input)
        output = self.bnOne(output)
        output = F.relu(output)

        output = self.conv2dTwo(output)
        output = self.bnTwo(output)
        output = F.relu(output)

        return output


class EAST(nn.Module):
    def __init__(self):
        super(EAST, self).__init__()
        # feature extractor stem: resnet50
        self.bbNet = pm.__dict__['resnet50'](pretrained='imagenet') # resnet50 in paper
        self.bbNet.eval()
        # backbone as feature extractor
        for param in self.bbNet.parameters():
            param.requires_grad = False

        # feature-merging branch
        self.mergeLayers0 = DummyLayer()
        self.mergeLayers1 = HLayer(2048 + 1024, 128)  # f:1024  g:2048 h:128
        self.mergeLayers2 = HLayer(128 + 512, 64)  # f:512 g:128 h:64
        self.mergeLayers3 = HLayer(64 + 256, 32)  # f:256 g:64 h:32

        self.mergeLayers4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32, momentum=0.003)

        # Output Layer
        self.textScale = 512
        self.scoreMap = nn.Conv2d(32, 1, kernel_size=1)
        self.geoMap = nn.Conv2d(32, 4, kernel_size=1)
        self.angleMap = nn.Conv2d(32, 1, kernel_size=1)

        # network weight parameters initial(conv, bn, linear)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_().fmod_(2).mul_(0.01).add_(0)
                # init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()

    def __foward_backbone(self, input):
        conv2 = None
        conv3 = None
        conv4 = None
        output = None  # batch_Size * 7 * 7 * 2048

        for name, layer in self.bbNet.named_children():
            input = layer(input)
            if name == 'layer1':
                conv2 = input
            elif name == 'layer2':
                conv3 = input
            elif name == 'layer3':
                conv4 = input
            elif name == 'layer4':
                output = input
                break

        return output, conv4, conv3, conv2

    def __unpool(self, input):
        _, _, H, W = input.shape
        return F.interpolate(input, mode='bilinear', scale_factor=2, align_corners=True)

    def __mean_image_subtraction(self, images, means=[123.68, 116.78, 103.94]):
        '''
        image normalization
        :param images: bs * w * h * channel
        :param means:
        :return:
        '''
        num_channels = images.data.shape[1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')
        for i in range(num_channels):
            images.data[:, i, :, :] -= means[i]

        return images

    def forward(self, input):

        input = self.__mean_image_subtraction(input)

        # bottom up

        f = self.__foward_backbone(input)

        g = [None] * 4
        h = [None] * 4

        # i = 1
        h[0] = self.mergeLayers0(f[0])  # f[0]:2048, h[0]:2048
        g[0] = self.__unpool(h[0])  # g[0]:2048

        # i = 2
        h[1] = self.mergeLayers1(g[0], f[1])  # g[0]=2048, f[1]=1024, h[1]=128
        g[1] = self.__unpool(h[1])  # g[1]=128

        # i = 3
        h[2] = self.mergeLayers2(g[1], f[2])  # g[1]=128, f[2]=512, h[2]=64
        g[2] = self.__unpool(h[2])  # g[2] = 64

        # i = 4
        h[3] = self.mergeLayers3(g[2], f[3])  # g[2]=64, f[3]=256, h[3]=32
        # g[3] = self.__unpool(h[3])

        # final stage
        final = self.mergeLayers4(h[3])  # 32
        final = self.bn5(final)
        final = F.relu(final)

        score = self.scoreMap(final)
        score = torch.sigmoid(score)

        geoMap = self.geoMap(final)
        geoMap = torch.sigmoid(geoMap) * self.textScale

        angleMap = self.angleMap(final)
        angleMap = (torch.sigmoid(angleMap) - 0.5) * math.pi / 2  # angle is between [-45, 45]
        geometry = torch.cat([geoMap, angleMap], dim=1)

        return score, geometry


