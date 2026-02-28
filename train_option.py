# -*- coding: utf-8 -*-
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DIR', default='../dataset/imagenet',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8) for one node')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--gpu', type=int, nargs="+", help='0 to set a single GPU, [0, 1, 2] to set different GPUs')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--num_crops', default=4, type=int,
                        help='number of crops in each image, 1 is the standard training')
    parser.add_argument('--softlabel_path', default='./FKD_soft_label_500_crops_marginal_smoothing_k_5', type=str, metavar='PATH',
                        help='path to soft label files (default: none)')
    parser.add_argument('--save_checkpoint_path', default='./FKD_checkpoints_output', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--soft_label_type', default='marginal_smoothing_k5', type=str, metavar='TYPE',
                        help='(1) ori; (2) hard; (3) smoothing; (4) marginal_smoothing_k5; (5) marginal_smoothing_k10; (6) marginal_renorm_k5')
    parser.add_argument('--num_classes', default=1000, type=int,
                        help='number of classes.')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--log', default='log_examp.log', type=str,
                        help='names of logging file.')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Vit parameters
    parser.add_argument('--model', default='deit_base_patch16_224_quant', type=str, metavar='MODEL',
                            help='Name of model to train')
    parser.add_argument('--wbits', default=-1, type=int)
    parser.add_argument('--abits', default=-1, type=int)
    parser.add_argument('--act-unsigned', action='store_true', default=False, help='using unsigned activations')
    parser.add_argument('--use-offset', action='store_true', default=False, help='using offset')
    parser.add_argument('--act-layer', default='gelu', type=str)
    parser.add_argument('--fixed-scale', action='store_true', default=False, help='use fixed scale and offset')
    parser.add_argument('--mixpre', action='store_true', default=False, help='whether to learn the bit-width')
    parser.add_argument('--bitops-scaler', type=float, default=0.)
    parser.add_argument('--show-bit-state', action='store_true', default=False)
    parser.add_argument('--head-wise', action='store_true', default=False)
    parser.add_argument('--reg', action='store_true', default=False)
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    return parser.parse_args()