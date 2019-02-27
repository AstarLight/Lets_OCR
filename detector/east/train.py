import torch
import os
from torch import nn
import sys
sys.path.append("./east_lib")
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import Net.net as Net
import Net.loss as Loss
import east_lib.data_utils as Lib
import time
import Config as cfg
import torch.backends.cudnn as cudnn


import warnings
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id

def val(test_loader, model, criterion, epoch):
    print("Start val...")
    losses = Lib.AverageMeter()

    for i, (img, score_map, geo_map, training_mask) in enumerate(test_loader):

        if cfg.gpu_id is not None:
            img, score_map, geo_map, training_mask = img.cuda(), score_map.cuda(), geo_map.cuda(), training_mask.cuda()

        f_score, f_geometry = model(img)
        loss = criterion(score_map, f_score, geo_map, f_geometry, training_mask)
        losses.update(loss.item(), img.size(0))

    print('EAST <==> VAL <==> Epoch: [epoch {0}] Loss {loss.val:.4f} Avg Loss {loss.avg:.4f})\n'.format(
        epoch, loss=losses))


def train(train_loader, test_loader, model, criterion, scheduler, optimizer, epoch):
    start = time.time()
    losses = Lib.AverageMeter()
    batch_time = Lib.AverageMeter()
    data_time = Lib.AverageMeter()
    end = time.time()
    model.train()

    for i, (img, score_map, geo_map, training_mask) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if cfg.gpu_id is not None:
            img, score_map, geo_map, training_mask = img.cuda(), score_map.cuda(), geo_map.cuda(), training_mask.cuda()

        f_score, f_geometry = model(img)
        loss = criterion(score_map, f_score, geo_map, f_geometry, training_mask)
        losses.update(loss.item(), img.size(0))

        # backward
        scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.display_iter == 0:
            print('EAST <==> TRAIN <==> Epoch: [{0}][{1}/{2}] Loss {loss.val:.4f} Avg Loss {loss.avg:.4f})\n'.format(
                epoch, i, len(train_loader), loss=losses))

        #save_loss_info(losses, epoch, i, train_loader)


if __name__ == "__main__":
    hmean = .0
    is_best = False

    if not os.path.exists(cfg.model_saved_path):
        os.mkdir(cfg.model_saved_path)

    warnings.simplefilter('ignore', np.RankWarning)
    # Prepare for dataset
    print('EAST <==> Prepare <==> DataLoader <==> Begin')
    train_root_path = cfg.data_path
    train_img = os.path.join(train_root_path, 'train_im')
    train_gt = os.path.join(train_root_path, 'train_gt')

    trainset = Lib.custom_dset(train_img, train_gt)
    train_loader = DataLoader(trainset, batch_size=cfg.batch_size,
                              shuffle=True, collate_fn=Lib.collate_fn, num_workers=cfg.num_workers)

    test_root_path = cfg.data_path
    test_img = os.path.join(test_root_path, 'test_im')
    test_gt = os.path.join(test_root_path, 'test_gt')

    testset = Lib.custom_dset(test_img, test_gt)
    test_loader = DataLoader(testset, batch_size=cfg.batch_size,
                             shuffle=True, collate_fn=Lib.collate_fn, num_workers=cfg.num_workers)

    print("total train img num: %s" % len(os.listdir(train_img)))
    print("total test img num: %s" % len(os.listdir(test_img)))

    print('EAST <==> Prepare <==> Batch_size:{} <==> Begin'.format(cfg.batch_size))
    print('EAST <==> Prepare <==> DataLoader <==> Done')


    # Model
    print('EAST <==> Prepare <==> Network <==> Begin')
    model = Net.EAST()
    print(model)
    #model = nn.DataParallel(model, device_ids=cfg.gpu_id)
    model = model.cuda()
    #init_weights(model, init_type=cfg.init_type)
    cudnn.benchmark = True

    criterion = Loss.EAST_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.94)

    # init or resume
    if cfg.resume and os.path.isfile(cfg.check_point):
        weightpath = os.path.abspath(cfg.check_point)
        print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Begin".format(weightpath))
        checkpoint = torch.load(weightpath)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Done".format(weightpath))
    else:
        start_epoch = 0
    print('EAST <==> Prepare <==> Network <==> Done')

    for epoch in range(start_epoch, cfg.epoch):

        train(train_loader, test_loader, model, criterion, scheduler, optimizer, epoch)

        if epoch % cfg.val_iter == 0:
            model.eval()
            print('Start evaluate at epoch {0} iteration.'.format(epoch))
            val(test_loader, model, criterion, epoch)
            model.train()

        if (epoch+1) % cfg.save_iter == 0:
            torch.save(model.state_dict(),
                       os.path.join(cfg.model_saved_path, 'east-msra_ali-{0}-end.model'.format(cfg.epoch)))

    torch.save(model.state_dict(), os.path.join(cfg.model_saved_path, 'final_east-msra_ali-{0}-end.model'.format(cfg.epoch)))
