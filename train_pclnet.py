from __future__ import print_function

import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import tensorboard_logger as tb_logger
from torchvision import transforms
from operations.util import adjust_learning_rate, AverageMeter, accuracy, ProgressMeter
from models.cnn_model import Net
from operations.InforNCE import MemoryBank, InforNCE
from data_collect import self_supervised_dataset


from config_para import parse_option_train
import numpy as np


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data = p2.data * m + p1.data * (1. - m)


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    args = parse_option_train()
    torch.set_num_threads(args.set_num_threads)
    train_dataset = self_supervised_dataset(args, args.data_folder, transform=transforms.ToTensor())
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler, drop_last=True)

    # create model and optimizer
    n_data = len(train_dataset)
    model = Net(args.outDim)
    # copy weights from `model' to `model_ema'
    model_ema = Net(args.outDim)

    # set the contrast memory and criterion
    contrast = MemoryBank(args.outDim, n_data, args.nce_k, args.nce_t, args.softmax).cuda()
    criterion = InforNCE()
    criterion = criterion.cuda()

    model = model.cuda()
    model_ema = model_ema.cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            print("=> loaded successfully '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    loss_save = np.zeros(args.epochs)
    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(optimizer, epoch, args)
        print("==> training...")

        time1 = time.time()
        loss = train_pclnet(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, args)
        loss_save[epoch-1]=loss
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }

            state['model_ema'] = model_ema.state_dict()
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state
        # saving the model
        print('==> Saving...')
        state = {
            'opt': args,
            'model': model.state_dict(),
            'contrast': contrast.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }

        state['model_ema'] = model_ema.state_dict()
        save_file = os.path.join(args.model_folder, 'current.pth')
        torch.save(state, save_file)
        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
        # help release GPU memory
        del state
        torch.cuda.empty_cache()


def train_pclnet(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, opt):

    model.train()
    model_ema.eval()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_meter = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_meter, top1],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for idx, inputs in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)

        inputs = inputs.float()
        inputs = inputs.cuda(non_blocking=True)

        # ===================forward=====================
        x1, x2 = torch.split(inputs, [9, 9], dim=1)

        feat_q = model(x1)
        with torch.no_grad():
            feat_k = model_ema(x2)

        out = contrast(feat_q, feat_k)

        loss, target = criterion(out)
        acc1 = accuracy(out, target)[0]
        # ===================backward=====================
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        top1.update(acc1[0], bsz)

        moment_update(model, model_ema, opt.alpha)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            progress.display(idx)
            sys.stdout.flush()

    return loss_meter.avg


if __name__ == '__main__':
    main()
