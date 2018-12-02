import torch.optim as optim
import torch
import cv2
import lib.tag_anchor
import lib.generate_gt_anchor
import lib.dataset_handler
import lib.utils
import numpy as np
import os
import Net.net as Net
import Net.loss as Loss
import ConfigParser
import time
import evaluate
import logging
import datetime
import copy
import random
import matplotlib.pyplot as plt

DRAW_PREFIX = './anchor_draw'
MSRA = '/home/ljs/data_ready/MSRA_TD500'
ALI = '/home/ljs/data_ready/ali_icpr'
DATASET_LIST = [MSRA, ALI]
MODEL_SAVE_PATH = '/model'


def loop_files(path):
    files = []
    l = os.listdir(path)
    for f in l:
        files.append(os.path.join(path, f))
    return files


def create_train_val():
    train_im_list = []
    test_im_list = []
    train_gt_list = []
    test_gt_list = []
    for dataset in DATASET_LIST:
        trains_im_path =os.path.join(dataset, 'train_im')
        tests_im_path = os.path.join(dataset, 'test_im')
        trains_gt_path =os.path.join(dataset, 'train_gt')
        test_gt_path = os.path.join(dataset, 'test_gt')
        train_im = loop_files(trains_im_path)
        train_gt = loop_files(trains_gt_path)
        test_im = loop_files(tests_im_path)
        test_gt = loop_files(test_gt_path)
        train_im_list += train_im
        test_im_list += test_im
        train_gt_list += train_gt
        test_gt_list += test_gt
    return train_im_list, train_gt_list, test_im_list, test_gt_list


def draw_loss_plot(train_loss_list=[], test_loss_list=[]):
    x1 = range(0, len(train_loss_list))
    x2 = range(0, len(test_loss_list))
    y1 = train_loss_list
    y2 = test_loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('train loss vs. iterators')
    plt.ylabel('train loss')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('test loss vs. iterators')
    plt.ylabel('test loss')
    plt.savefig("test_train_loss.jpg")


if __name__ == '__main__':
    cf = ConfigParser.ConfigParser()
    cf.read('./config')

    log_dir = './logs_10'

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    log_file_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    log_handler = logging.FileHandler(os.path.join(log_dir, log_file_name), 'w')
    log_format = formatter = logging.Formatter('%(asctime)s: %(message)s')
    log_handler.setFormatter(log_format)
    logger.addHandler(log_handler)

    gpu_id = cf.get('global', 'gpu_id')
    epoch = cf.getint('global', 'epoch')
    val_batch_size = cf.getint('global', 'val_batch')
    logger.info('Total epoch: {0}'.format(epoch))

    using_cuda = cf.getboolean('global', 'using_cuda')
    display_img_name = cf.getboolean('global', 'display_file_name')
    display_iter = cf.getint('global', 'display_iter')
    val_iter = cf.getint('global', 'val_iter')
    save_iter = cf.getint('global', 'save_iter')

    lr_front = cf.getfloat('parameter', 'lr_front')
    lr_behind = cf.getfloat('parameter', 'lr_behind')
    change_epoch = cf.getint('parameter', 'change_epoch') - 1
    logger.info('Learning rate: {0}, {1}, change epoch: {2}'.format(lr_front, lr_behind, change_epoch + 1))
    print('Using gpu id(available if use cuda): {0}'.format(gpu_id))
    print('Train epoch: {0}'.format(epoch))
    print('Use CUDA: {0}'.format(using_cuda))

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    no_grad = [
        'cnn.VGG_16.convolution1_1.weight',
        'cnn.VGG_16.convolution1_1.bias',
        'cnn.VGG_16.convolution1_2.weight',
        'cnn.VGG_16.convolution1_2.bias'
    ]

    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)

    net = Net.CTPN()
    for name, value in net.named_parameters():
        if name in no_grad:
            value.requires_grad = False
        else:
            value.requires_grad = True
    # for name, value in net.named_parameters():
    #     print('name: {0}, grad: {1}'.format(name, value.requires_grad))
    net.load_state_dict(torch.load('./lib/vgg16.model'))
    # net.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    lib.utils.init_weight(net)
    if using_cuda:
        net.cuda()
    net.train()
    print(net)

    criterion = Loss.CTPN_Loss(using_cuda=using_cuda)

    train_im_list, train_gt_list, val_im_list, val_gt_list = create_train_val()
    total_iter = len(train_im_list)
    print("total training image num is %s" % len(train_im_list))
    print("total val image num is %s" % len(val_im_list))

    train_loss_list = []
    test_loss_list = []

    for i in range(epoch):
        if i >= change_epoch:
            lr = lr_behind
        else:
            lr = lr_front
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        #optimizer = optim.Adam(net.parameters(), lr=lr)
        iteration = 1
        total_loss = 0
        total_cls_loss = 0
        total_v_reg_loss = 0
        total_o_reg_loss = 0
        start_time = time.time()

        random.shuffle(train_im_list)
        # print(random_im_list)
        for im in train_im_list:
            root, file_name = os.path.split(im)
            root, _ = os.path.split(root)
            name, _ = os.path.splitext(file_name)
            gt_name = 'gt_' + name + '.txt'

            gt_path = os.path.join(root, "train_gt", gt_name)

            if not os.path.exists(gt_path):
                print('Ground truth file of image {0} not exists.'.format(im))
                continue

            gt_txt = lib.dataset_handler.read_gt_file(gt_path)
            #print("processing image %s" % os.path.join(img_root1, im))
            img = cv2.imread(im)
            if img is None:
                iteration += 1
                continue

            img, gt_txt = lib.dataset_handler.scale_img(img, gt_txt)
            tensor_img = img[np.newaxis, :, :, :]
            tensor_img = tensor_img.transpose((0, 3, 1, 2))
            if using_cuda:
                tensor_img = torch.FloatTensor(tensor_img).cuda()
            else:
                tensor_img = torch.FloatTensor(tensor_img)

            vertical_pred, score, side_refinement = net(tensor_img)
            del tensor_img

            # transform bbox gt to anchor gt for training
            positive = []
            negative = []
            vertical_reg = []
            side_refinement_reg = []

            visual_img = copy.deepcopy(img)

            try:
                # loop all bbox in one image
                for box in gt_txt:
                    # generate anchors from one bbox
                    gt_anchor, visual_img = lib.generate_gt_anchor.generate_gt_anchor(img, box, draw_img_gt=visual_img)
                    positive1, negative1, vertical_reg1, side_refinement_reg1 = lib.tag_anchor.tag_anchor(gt_anchor, score, box)
                    positive += positive1
                    negative += negative1
                    vertical_reg += vertical_reg1
                    side_refinement_reg += side_refinement_reg1
            except:
                print("warning: img %s raise error!" % im)
                iteration += 1
                continue

            if len(vertical_reg) == 0 or len(positive) == 0 or len(side_refinement_reg) == 0:
                iteration += 1
                continue

            cv2.imwrite(os.path.join(DRAW_PREFIX, file_name), visual_img)
            optimizer.zero_grad()
            loss, cls_loss, v_reg_loss, o_reg_loss = criterion(score, vertical_pred, side_refinement, positive,
                                                               negative, vertical_reg, side_refinement_reg)
            loss.backward()
            optimizer.step()
            iteration += 1
            # save gpu memory by transferring loss to float
            total_loss += float(loss)
            total_cls_loss += float(cls_loss)
            total_v_reg_loss += float(v_reg_loss)
            total_o_reg_loss += float(o_reg_loss)

            if iteration % display_iter == 0:
                end_time = time.time()
                total_time = end_time - start_time
                print('Epoch: {2}/{3}, Iteration: {0}/{1}, loss: {4}, cls_loss: {5}, v_reg_loss: {6}, o_reg_loss: {7}, {8}'.
                      format(iteration, total_iter, i, epoch, total_loss / display_iter, total_cls_loss / display_iter,
                             total_v_reg_loss / display_iter, total_o_reg_loss / display_iter, im))

                logger.info('Epoch: {2}/{3}, Iteration: {0}/{1}'.format(iteration, total_iter, i, epoch))
                logger.info('loss: {0}'.format(total_loss / display_iter))
                logger.info('classification loss: {0}'.format(total_cls_loss / display_iter))
                logger.info('vertical regression loss: {0}'.format(total_v_reg_loss / display_iter))
                logger.info('side-refinement regression loss: {0}'.format(total_o_reg_loss / display_iter))

                train_loss_list.append(total_loss)

                total_loss = 0
                total_cls_loss = 0
                total_v_reg_loss = 0
                total_o_reg_loss = 0
                start_time = time.time()

            if iteration % val_iter == 0:
                net.eval()
                logger.info('Start evaluate at {0} epoch {1} iteration.'.format(i, iteration))
                val_loss = evaluate.val(net, criterion, val_batch_size, using_cuda, logger, val_im_list)
                logger.info('End evaluate.')
                net.train()
                start_time = time.time()
                test_loss_list.append(val_loss)

            if iteration % save_iter == 0:
                print('Model saved at ./model/ctpn-{0}-{1}.model'.format(i, iteration))
                torch.save(net.state_dict(), os.path.join(MODEL_SAVE_PATH, 'ctpn-msra_ali-{0}-{1}.model'.format(i, iteration)))

        print('Model saved at ./model/ctpn-{0}-end.model'.format(i))
        torch.save(net.state_dict(), os.path.join(MODEL_SAVE_PATH, 'ctpn-msra_ali-{0}-end.model'.format(i)))

        draw_loss_plot(train_loss_list, test_loss_list)
