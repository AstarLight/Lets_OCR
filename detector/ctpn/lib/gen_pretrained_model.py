import torch
import torchvision.models as models
import sys
sys.path.append('..')
import Net


def generate_VGG_16_model(net, output_path):
    vgg_16 = models.vgg16(pretrained=False)
    print(vgg_16)
    vgg_16.load_state_dict(torch.load('./vgg16-397923af.pth'))
    pretrained_dict = vgg_16.state_dict()
    model_dict = net.state_dict()
    check_list = [[], []]
    for i in range((2 + 2 + 3 + 3 + 3) * 2):
        check_list[0].append(pretrained_dict.keys()[i])
        check_list[1].append(model_dict.keys()[i])
    backbone_dict = {}
    for j in range((2 + 2 + 3 + 3 + 3) * 2):
        backbone_dict[check_list[1][j]] = pretrained_dict[check_list[0][j]]
    model_dict.update(backbone_dict)
    # check model
    # for k in range((2 + 2 + 3 + 3 + 3) * 2):
    #     print((model_dict[model_dict.keys()[k]] == pretrained_dict[pretrained_dict.keys()[k]]).all())
    torch.save(model_dict, output_path)


if __name__ == '__main__':
    generate_VGG_16_model(Net.CTPN(), './vgg16.model')