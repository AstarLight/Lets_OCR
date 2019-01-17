import Dataset
import Config
import torch


def val(net, dataset, criterion, converter, image, text, length, max_iter=10):
    print('Start val')

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=Config.batch_size, num_workers=int(Config.data_worker),
        collate_fn=Dataset.alignCollate(imgH=Config.img_height, imgW=Config.img_width, keep_ratio=True))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = Dataset.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        Dataset.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        Dataset.loadData(text, t)
        Dataset.loadData(length, l)

        preds = net(image)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        # preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target:
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:Config.test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * Config.batch_size)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))