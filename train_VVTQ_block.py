import argparse
import os
import random
import shutil
import time
import math
import warnings
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
# 引用当前目录下的FKD-main/FKD/FKD_ViT/目录中的utils_FKD.py文件
current_dir = os.path.dirname(os.path.abspath(__file__))
fkdvit_path = os.path.join(current_dir, 'FKD-main', 'FKD', 'FKD_ViT')
sys.path.append(fkdvit_path)

from utils_FKD import RandomResizedCrop_FKD, RandomHorizontalFlip_FKD, ImageFolder_FKD, Compose_FKD, Soft_CrossEntropy, Recover_soft_label
from torchvision.transforms import InterpolationMode
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.models import create_model
#from models.mobilenet_imagenet_pact import *
#from models.mobilenet_imagenet_prelu import *
from quantization import DeiT_quant, SReT_quant, Swin_quant
from engine import initialize_quantization
from util_loss import BinReg, CosineTempDecay
from utils import *
import torch.utils.tensorboard as tensorboard
from pathlib import Path
from train_option import get_args_parser
import pickle

# timm is used to build the optimizer and learning rate scheduler (https://github.com/rwightman/pytorch-image-models)

best_acc1 = 0

def main(args):
    if not os.path.exists(args.save_checkpoint_path):
        output_dir = Path(args.save_checkpoint_path)
        output_dir.mkdir(parents=True, exist_ok=True)

    # convert to TRUE number of loading-images and #epochs since we use multiple crops from the same image within a minbatch
    args.batch_size = math.ceil(args.batch_size / args.num_crops)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.world_size = len(args.gpu)
    args.distributed = args.world_size > 1

    if args.distributed:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, args))
    else:
        # Simply call main_worker function
        main_worker(0, args.world_size, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.rank = int(gpu)
    print(f'args.rank: {args.rank}')
    output_dir = Path(args.save_checkpoint_path)

    tb_writer = None
    logger = None
    if args.rank == 0:        
        logger = get_logger(args.save_checkpoint_path, name='train')
        logger.info(args)
        tb_writer = tensorboard.SummaryWriter(log_dir=args.save_checkpoint_path)

    if args.distributed:
        dist.init_process_group("nccl", world_size=args.world_size, rank=args.rank)
    # create model
    if args.rank == 0: logger.info(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        wbits=args.wbits,
        abits=args.abits,
        act_layer=nn.GELU,
        offset=args.use_offset,
        learned=not args.fixed_scale,
        mixpre=args.mixpre,
        headwise=args.head_wise
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False, pickle_module=pickle)

        checkpoint_model = checkpoint['model']
        model.load_state_dict(checkpoint_model, strict=False)

    if not torch.cuda.is_available():
        if args.rank == 0: logger.info('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(args.rank)
        model.cuda(args.rank)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.rank == 0:
            logger.info(f"update batch_size: {args.batch_size}")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank])

        # if args.gpu is not None:
        #     torch.cuda.set_device(args.gpu)
        #     model.cuda(args.gpu)
        #     # When using a single GPU per process and per
        #     # DistributedDataParallel, we need to divide the batch size
        #     # ourselves based on the total number of GPUs we have
        #     args.batch_size = int(args.batch_size / ngpus_per_node)
        #     args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # else:
        #     model.cuda()
        #     # DistributedDataParallel will divide and allocate batch_size to all
        #     # available GPUs if device_ids are not set
        #     model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.rank is not None:
        torch.cuda.set_device(args.rank)
        model = model.cuda(args.rank)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion_sce = Soft_CrossEntropy()
    criterion_ce = nn.CrossEntropyLoss().cuda(args.rank)

    optimizer = create_optimizer(args, model)

    lr_scheduler, _ = create_scheduler(args, optimizer)

    # optionally resume from a checkpointresume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.rank is None:
                checkpoint = torch.load(args.resume, weights_only=False, pickle_module=pickle)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.rank)
                checkpoint = torch.load(args.resume, map_location=loc, weights_only=False, pickle_module=pickle)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.rank is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.rank)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_dataset = ImageFolder_FKD(
        num_crops=args.num_crops,
        softlabel_path=args.softlabel_path,
        root=traindir,
        transform=Compose_FKD(transforms=[
            RandomResizedCrop_FKD(size=224,
                                  interpolation='bilinear'), 
            RandomHorizontalFlip_FKD(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_dataset_single_crop = ImageFolder_FKD(
        num_crops=1,
        softlabel_path=args.softlabel_path,
        root=traindir,
        transform=Compose_FKD(transforms=[
            RandomResizedCrop_FKD(size=224,
                                  interpolation='bilinear'), 
            RandomHorizontalFlip_FKD(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    train_loader_single_crop = torch.utils.data.DataLoader(
        train_dataset_single_crop, batch_size=args.batch_size*args.num_crops, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    dataset_val = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    data_loader_sampler = torch.utils.data.DataLoader(
        dataset_val, sampler=torch.utils.data.SequentialSampler(dataset_val),
        batch_size=64,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )

    device = torch.device('cuda')
    # 为避免过度打印，在每个epoch中最多打印10次
    train_loader_len = len(train_loader)
    args.print_freq = train_loader_len // 10 if train_loader_len // 10 > 0 else args.print_freq
    if args.rank == 0: logger.info(f"train_loader_len: {train_loader_len}, print_freq: {args.print_freq}")

    # # TODO 先eval下fp模型
    # if not args.resume:
    #     if args.rank == 0: logger.info("Starting evaluation of FP model..")
    #     validate(val_loader, model, criterion_ce, args, 0, tb_writer, logger=logger)
    #     if args.rank == 0: logger.info("end evaluation of FP model")

    if args.resume == '' and (args.abits > 0 or args.wbits > 0):
        if args.rank == 0: logger.info("Starting quantization scale initialization")
        initialize_quantization(data_loader_sampler, model, device, output_dir, sample_iters=5, logger=logger)

    if args.evaluate:
        validate(val_loader, model, criterion_ce, args, 0, tb_writer, logger=logger)
        return

    # warmup with single crop, "=" is used to let start_epoch to be 0 for the corner case.
    if args.start_epoch <= args.warmup_epochs:
        for epoch in range(args.start_epoch, args.warmup_epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            # train for one epoch
            train(train_loader_single_crop, model, criterion_sce, optimizer, epoch, args, tb_writer, logger=logger)
            lr_scheduler.step(epoch + 1)

            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion_ce, args, epoch, tb_writer, logger=logger)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if args.rank == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, filename=args.save_checkpoint_path+'/checkpoint.pth.tar')
        args.start_epoch = 0 # for resume
    else:
        args.warmup_epochs = args.num_crops - 1  # for resume

    for epoch in range(args.start_epoch+args.warmup_epochs, args.epochs, args.num_crops):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # for fine-grained evaluation at last a few epochs
        if epoch >= (args.epochs-args.num_crops):
            start_epoch = epoch
            for epoch in range(start_epoch, args.epochs):
                if args.distributed:
                    train_sampler.set_epoch(epoch)
                lr_scheduler.step(epoch+1)
                # train for one epoch
                train(train_loader_single_crop, model, criterion_sce, optimizer, epoch, args, tb_writer, logger=logger)

                # evaluate on validation set
                acc1 = validate(val_loader, model, criterion_ce, args, epoch, tb_writer, logger=logger)

                # remember best acc@1 and save checkpoint
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)

                if args.rank == 0:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, filename=args.save_checkpoint_path+'/checkpoint.pth.tar')
            return 
        else:
            # train for one epoch
            train(train_loader, model, criterion_sce, optimizer, epoch, args, tb_writer, logger=logger)
        lr_scheduler.step(epoch+args.num_crops)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion_ce, args, epoch, tb_writer, logger=logger)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename=args.save_checkpoint_path+'/checkpoint.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch, args, tb_writer, logger=None):
    batch_time = AverageMeter('Time', ':.3f')
    data_time = AverageMeter('Data', ':.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':.2f')
    top5 = AverageMeter('Acc@5', ':.2f')
    if args.reg:
        annealing_schedule = CosineTempDecay(t_max=args.epochs, temp_range=(0, 0.01), rel_decay_start=0.0)
        annealing_schedule_reg = annealing_schedule(epoch)

        oscreg = AverageMeter('regLoss', ':.4f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, oscreg, losses, top1, top5, 'LR {lr:.5f}'.format(lr=_get_learning_rate(optimizer)), 'regloss_s {reg:.5f}'.format(reg=annealing_schedule_reg)],
            prefix="Epoch: [{}]".format(epoch), logger=logger)
    else:
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5, 'LR {lr:.5f}'.format(lr=_get_learning_rate(optimizer))],
            prefix="Epoch: [{}]".format(epoch), logger=logger)

    # switch to train mode
    model.train()
    end = time.time()
    
    for i, (images, target, soft_label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = torch.cat(images, dim=0)
        soft_label = torch.cat(soft_label, dim=0)
        target = torch.cat(target, dim=0)

        if args.soft_label_type != 'ori':
            soft_label = Recover_soft_label(soft_label, args.soft_label_type, args.num_classes)

        if args.rank is not None:
            images = images.cuda(args.rank, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.rank, non_blocking=True)
            soft_label = soft_label.cuda(args.rank, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, soft_label)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # add regularization
        if args.reg:
            bin_regularizer = BinReg(annealing_schedule_reg)
            loss_reg = bin_regularizer(model)
            loss += loss_reg
            oscreg.update(loss_reg.item(), images.size(0))

        # compute gradient and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank == 0 and i % args.print_freq == 0:
            progress.display(i)
    if args.rank == 0:
        tb_writer.add_scalar('train/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/acc1', top1.avg, epoch)
        tb_writer.add_scalar('train/acc5', top5.avg, epoch)
        if args.reg:
            tb_writer.add_scalar('train/reg_loss', oscreg.avg, epoch)
            tb_writer.add_scalar('train/regloss_s', annealing_schedule_reg, epoch)
        tb_writer.add_scalar('train/lr', _get_learning_rate(optimizer), epoch)


def validate(val_loader, model, criterion, args, epoch, tb_writer, logger=None):
    batch_time = AverageMeter('Time', ':.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':.2f')
    top5 = AverageMeter('Acc@5', ':.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ', logger=logger)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.rank is not None:
                images = images.cuda(args.rank, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.rank, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        if args.rank == 0:
            logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
            tb_writer.add_scalar('val/loss', losses.avg, epoch)
            tb_writer.add_scalar('val/acc1', top1.avg, epoch)
            tb_writer.add_scalar('val/acc5', top5.avg, epoch)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-19]+'/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}:{val' + self.fmt + '}({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.logger is not None: 
            self.logger.info(' '.join(entries))
        else:
            print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def _get_learning_rate(optimizer):
    return max(param_group['lr'] for param_group in optimizer.param_groups)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    args = get_args_parser()
    
    # 设置 CUDA 可见设备
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))
    print("train devices={}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    args.world_size = len(args.gpu)

    # 设置分布式训练的主地址和端口
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = find_free_port()
    main(args)