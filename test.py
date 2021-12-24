import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import random
import matplotlib.pyplot as plt

from torch.autograd import Variable
from architect import Architect
from resnet import *
from model_search import Network
from classification_head import *
from dataset import *
from operations import *
from genotypes import PRIMITIVES
from genotypes import Genotype
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float,
                    default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--gpu',  type=str, default='0,1',
                    help='gpu device id')
parser.add_argument('--epochs', type=int, default=50,
                    help='num of training epochs')
parser.add_argument('--init_channels', type=int,
                    default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8,
                    help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true',
                    default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int,
                    default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float,
                    default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float,
                    default=0.7, help='portion of training data')
parser.add_argument('--unrolled', action='store_true',
                    default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float,
                    default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float,
                    default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--source', type=str,
                    default='amazon', help='Source Domain')
parser.add_argument('--target', type=str, default='dslr', help='Target Domain')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

DOMAIN_CLASSES = 2
NUM_CLASSES = 31
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():

    np.random.seed(args.seed)
    # torch.cuda.set_device(args.gpu)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    output_dim = args.init_channels*16
    domain_model = Network(
        args.init_channels, DOMAIN_CLASSES, args.layers, criterion)
    general_model = ResNet(criterion, output_dim)
    classification_head = ClassificationHead(output_dim, NUM_CLASSES)
    classification_head = classification_head.cuda()

    '''classification_head = nn.Sequential(
        nn.Linear(256, 31),
    )'''

    print("GPUS: ", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        gpus = [int(i) for i in args.gpu.split(',')]
        print(gpus, gpus[0])
        domain_model = nn.parallel.DataParallel(
            domain_model, device_ids=gpus, output_device=gpus[0]).cuda()
        general_model = nn.parallel.DataParallel(
            general_model, device_ids=gpus, output_device=gpus[0]).cuda()
        classification_head = nn.parallel.DataParallel(
            classification_head, device_ids=gpus, output_device=gpus[0]).cuda()

        domain_model = domain_model.module
        general_model = general_model.module
        classification_head = classification_head.module

    else:
        domain_model = domain_model.cuda()
        general_model = general_model.cuda()
        classification_head = classification_head.cuda()

    logging.info("Domain param size = %fMB",
                 utils.count_parameters_in_MB(domain_model))
    logging.info("Class param size = %fMB",
                 utils.count_parameters_in_MB(general_model))
    logging.info("Classification Head param size = %fMB",
                 utils.count_parameters_in_MB(classification_head))

    optimizer = torch.optim.SGD(domain_model.parameters(
    ), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_general = torch.optim.SGD(general_model.parameters(
    ), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_class_head = torch.optim.SGD(classification_head.parameters(
    ), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    source_train_queue, source_val_queue = get_dataloader(get_office_dataset(
        args.source, args.data), args.batch_size, num_workers=0, train_ratio=args.train_portion)
    target_train_queue, target_valid_queue = get_dataloader(get_office_dataset(
        args.target, args.data), args.batch_size, num_workers=0, train_ratio=args.train_portion)

    print("source_train_queue: ", len(source_train_queue),
          " source_val_queue: ", len(source_val_queue))
    print("target_train_queue: ", len(target_train_queue),
          " target_valid_queue: ", len(target_valid_queue))
    for step, (input, target) in enumerate(source_val_queue):
        input = torch.squeeze(input)
        print(target)
        plt.title(target)
        plt.imshow(input.permute(1, 2, 0))
        plt.show()
        print("=================")


if __name__ == '__main__':
    main()
