import argparse
import torch
import logging

import traceback
import Net.net as Net
from east_lib.bbox import Toolbox
import os


using_gpu = True

def load_model(model_path):
    logger.info("Loading checkpoint: {} ...".format(model_path))
    model = Net.EAST()
    if using_gpu:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    if using_gpu:
        model = model.cuda()
    model.eval()
    return model


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Model eval')
    parser.add_argument('-m', '--model', default="save_model/model_5.pth", type=str,
                        help='path to model')
    parser.add_argument('-o', '--output_dir', default="test_result/", type=str,
                        help='output dir for drawn images')
    parser.add_argument('-i', '--input_dir', default="test_pic/", type=str, required=False,
                        help='dir for input images')
    args = parser.parse_args()

    model = load_model(args["model"])

    images_root = args["input_dir"]
    with_image = True if args["output_dir"] else False  # save draw predict images
    for fn in os.listdir(images_root):
        full_path = os.path.join(images_root, fn)
        polys, im = Toolbox.predict(full_path, model, with_image, args["output_dir"], using_gpu)


