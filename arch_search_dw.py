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

parser = argparse.ArgumentParser("o31")
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
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

    output_dim = args.init_channels*16
    domain_model = Network(
        args.init_channels, DOMAIN_CLASSES, args.layers, criterion)
    general_model = ResNet(criterion, output_dim)
    classification_head = ClassificationHead(output_dim, NUM_CLASSES)
    classification_head = classification_head.cuda()

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

    train_transform, valid_transform = utils._data_transforms_office31(
        args)
    print("D-W")
    source_data_train = dset.ImageFolder(
        '../newdata/dslr/images/train', transform=train_transform)
    source_data_val = dset.ImageFolder(
        '../newdata/dslr/images/val', transform=valid_transform)
    target_data_train = dset.ImageFolder(
        '../newdata/webcam/images/train', transform=train_transform)
    target_data_val = dset.ImageFolder(
        '../newdata/webcam/images/val', transform=valid_transform)
    print("source_data_train: ", len(source_data_train),
          "source_data_val: ", len(source_data_val),
          " target_data_train: ", len(target_data_train),
          " target_data_val", len(target_data_val),)

    source_train_queue = torch.utils.data.DataLoader(source_data_train, batch_size=args.batch_size,
                                                     shuffle=True,
                                                     pin_memory=True,
                                                     num_workers=2)
    source_val_queue = torch.utils.data.DataLoader(source_data_val, batch_size=args.batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=2)

    target_train_queue = torch.utils.data.DataLoader(target_data_train, batch_size=args.batch_size,
                                                     shuffle=True,
                                                     pin_memory=True,
                                                     num_workers=2)
    target_valid_queue = torch.utils.data.DataLoader(target_data_val, batch_size=args.batch_size,
                                                     shuffle=True,
                                                     pin_memory=True,
                                                     num_workers=2)
    print("source_train_queue: ", len(source_train_queue),
          " source_val_queue: ", len(source_val_queue))
    print("target_train_queue: ", len(target_train_queue),
          " target_valid_queue: ", len(target_valid_queue))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(domain_model, general_model,
                          classification_head, args)
    best_valid_acc_source = 0.0
    best_valid_acc_target = 0.0
    best_train_acc_source = 0.0
    best_train_acc_target = 0.0
    best_eval_res = 0.0
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = domain_model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(domain_model.alphas_normal, dim=-1))
        print(F.softmax(domain_model.alphas_reduce, dim=-1))
        print("---------------------------")

        # training
        train_acc_source, train_acc_target, train_obj = train(source_train_queue, source_val_queue, target_train_queue, target_valid_queue, domain_model,
                                                              general_model, classification_head, architect, criterion, optimizer, optimizer_general, optimizer_class_head, lr, epoch)
        logging.info('train_acc %f %f', train_acc_source, train_acc_target)
        writer.flush()

        # validation
        valid_acc_source, valid_acc_target, valid_obj = infer(
            source_val_queue, target_valid_queue, domain_model, general_model, classification_head, criterion, epoch)
        logging.info('valid_acc %f', valid_acc_source, valid_acc_target)
        writer.flush()
        eval_res = (valid_acc_source + valid_acc_target)/2
        writer.close()

        if train_acc_source > best_train_acc_source:
            best_train_acc_source = train_acc_source
        logging.info('train_acc_source %f, best_train_acc_source %f',
                     train_acc_source, best_train_acc_source)

        if train_acc_target > best_train_acc_target:
            best_train_acc_target = train_acc_target
        logging.info('train_acc_target %f, best_train_acc_target %f',
                     train_acc_target, best_train_acc_target)

        if valid_acc_source > best_valid_acc_source:
            best_valid_acc_source = valid_acc_source
        logging.info('valid_acc_source %f, best_valid_acc_source %f',
                     valid_acc_source, best_valid_acc_source)

        if valid_acc_target > best_valid_acc_target:
            best_valid_acc_target = valid_acc_target
        logging.info('valid_acc_target %f, best_valid_acc_target %f',
                     valid_acc_target, best_valid_acc_target)

        if eval_res > best_eval_res:
            best_eval_res = eval_res
        logging.info('eval_res %f, best_eval_res %f',
                     eval_res, best_eval_res)

        utils.save(domain_model, os.path.join(args.save, 'domain_weights.pt'))
        utils.save(general_model, os.path.join(
            args.save, 'general_weights.pt'))
        utils.save(classification_head, os.path.join(
            args.save, 'classification_head.pt'))

    genotype = domain_model.genotype()
    logging.info('genotype = %s', genotype)


def train(source_train_queue, source_val_queue, target_train_queue, target_valid_queue, domain_model,
          general_model, classification_head, architect, criterion, optimizer, optimizer_general, optimizer_class_head, lr, epoch):
    objs = utils.AvgrageMeter()
    objs1 = utils.AvgrageMeter()
    source_top1 = utils.AvgrageMeter()
    source_top5 = utils.AvgrageMeter()
    target_top1 = utils.AvgrageMeter()
    target_top5 = utils.AvgrageMeter()

    # print("1---------------------------")
    min_dataloader = min(len(source_train_queue), len(target_train_queue))
    data_source_iter = iter(source_train_queue)
    data_target_iter = iter(target_train_queue)
    count = 0
    #n_total = 0
    #n_correct_source = 0
    #n_correct_target = 0
    while count < min_dataloader:
        # print("2---------------------------")
        domain_model.train()

        data_source = data_source_iter.next()
        source_input, source_target = data_source
        #print("Input shape: ", source_input.shape)
        # print("Target shape: ", source_target.shape)
        # print("Domain shape: ", source_domain.shape)
        source_input = source_input.cuda()
        source_target = source_target.cuda(non_blocking=True)
        source_domain = torch.ones(source_input.size(0), dtype=torch.long)
        source_domain = source_domain.cuda(non_blocking=True)

        general_rep_soruce = general_model(source_input)
        _, logits_source_domain = domain_model(source_input)
        source_domain_loss = criterion(logits_source_domain, source_domain)

        data_target = data_target_iter.next()
        target_input, target_target = data_target
        target_input = target_input.cuda()
        target_target = target_target.cuda(non_blocking=True)
        print(source_target, target_target)
        target_domain = torch.zeros(target_input.size(0), dtype=torch.long)
        target_domain = target_domain.cuda(non_blocking=True)
        general_rep_target = general_model(target_input)
        _, logits_target_domain = domain_model(target_input)
        target_domain_loss = criterion(logits_target_domain, target_domain)
        n = source_input.size(0)

        input_search_source, target_search_source = next(
            iter(source_val_queue))
        input_search_source = input_search_source.cuda()
        target_search_source = target_search_source.cuda(non_blocking=True)
        input_search_target, target_search_target = next(
            iter(target_valid_queue))
        input_search_target = input_search_target.cuda()
        target_search_target = target_search_target.cuda(non_blocking=True)

        total_domain_loss = source_domain_loss + target_domain_loss

        optimizer.zero_grad()
        total_domain_loss.backward()
        nn.utils.clip_grad_norm(domain_model.parameters(), args.grad_clip)
        optimizer.step()

        domain_rep_source, _ = domain_model(source_input)
        domain_rep_target, _ = domain_model(target_input)

        class_rep_source = general_rep_soruce - domain_rep_source
        class_rep_target = general_rep_target - domain_rep_target
        logits_class_source = classification_head(class_rep_source)
        logits_class_target = classification_head(class_rep_target)
        source_class_loss = criterion(logits_class_source, source_target)
        target_class_loss = criterion(logits_class_target, target_target)

        total_class_loss = source_class_loss+target_class_loss
        optimizer_general.zero_grad()
        optimizer_class_head.zero_grad()
        total_class_loss.backward()
        optimizer_general.step()
        optimizer_class_head.step()
        # print(logits_class_source.shape)
        # print(source_target.shape)
        architect.step(source_input, source_target, source_domain, target_input, target_target, target_domain, input_search_source, target_search_source, source_domain, input_search_target, target_search_target, target_domain,
                       lr, optimizer, optimizer_general, optimizer_class_head, args.init_channels*16, unrolled=args.unrolled)
        architect.step1(source_input, source_target, source_domain, target_input, target_target, target_domain, input_search_source, target_search_source, source_domain, input_search_target, target_search_target, target_domain,
                        lr, optimizer, optimizer_general, optimizer_class_head, args.init_channels*16, unrolled=args.unrolled)
        architect.step2(source_input, source_target, source_domain, target_input, target_target, target_domain, input_search_source, target_search_source, source_domain, input_search_target, target_search_target, target_domain,
                        lr, optimizer, optimizer_general, optimizer_class_head, args.init_channels*16, unrolled=args.unrolled)

        '''pred_s = logits_class_source.data.max(1, keepdim=True)[1]
        n_correct_source += pred_s.eq(
            source_target.data.view_as(pred_s)).sum().item()
        n_total += args.batch_size
        pred_t = logits_class_target.data.max(1, keepdim=True)[1]
        n_correct_target += pred_t.eq(
            target_target.data.view_as(pred_t)).sum().item()'''

        source_prec1, source_prec5 = utils.accuracy(
            logits_class_source, source_target, topk=(1, 5))
        target_prec1, target_prec5 = utils.accuracy(
            logits_class_target, target_target, topk=(1, 5))

        objs.update(total_domain_loss.item(), n)
        objs1.update(total_class_loss.item(), n)
        source_top1.update(source_prec1.item(), n)
        source_top5.update(source_prec5.item(), n)
        target_top1.update(target_prec1.item(), n)
        target_top5.update(target_prec5.item(), n)

        if count % args.report_freq == 0:
            logging.info('train %03d %e %e %f %f', count, objs.avg,
                         objs1.avg, source_top1.avg, target_top1.avg)

        '''if count % args.report_freq == 0:
            logging.info('train %03d %e %e', count, source_class_loss,
                         source_class_loss)'''
        count += 1
    #acc_source = n_correct_source * 1.0 / n_total
    #acc_target = n_correct_target * 1.0 / n_total

    writer.add_scalar("Domain loss/train", objs.avg, epoch)
    writer.add_scalar("Class loss/train", objs1.avg, epoch)
    writer.add_scalar("Top1/train/source", source_top1.avg, epoch)
    writer.add_scalar("Top5/train/source", source_top5.avg, epoch)
    writer.add_scalar("Top1/train/target", target_top1.avg, epoch)
    writer.add_scalar("Top5/train/target", target_top5.avg, epoch)

    '''writer.add_scalar("Domain loss/train", total_domain_loss, epoch)
    writer.add_scalar("Class loss/train", total_class_loss, epoch)
    writer.add_scalar("Acc/train/source", acc_source, epoch)
    writer.add_scalar("Acc/train/target", acc_target, epoch)'''

    return source_top1.avg, target_top1.avg, objs1.avg


def infer(source_val_queue, target_valid_queue, domain_model, general_model, classification_head, criterion, epoch):
    objs = utils.AvgrageMeter()
    objs1 = utils.AvgrageMeter()
    source_top1 = utils.AvgrageMeter()
    source_top5 = utils.AvgrageMeter()
    target_top1 = utils.AvgrageMeter()
    target_top5 = utils.AvgrageMeter()

    min_dataloader = min(len(source_val_queue), len(target_valid_queue))
    data_source_iter = iter(source_val_queue)
    data_target_iter = iter(target_valid_queue)
    count = 0
    #res = {}
    #res['probs'], res['gt'] = [], []
    domain_model.eval()
    with torch.no_grad():
        #n_total = 0
        #n_correct_target = 0
        #n_correct_source = 0
        while count < min_dataloader:
            data_source = data_source_iter.next()
            source_input, source_target = data_source
            source_input = source_input.cuda()
            source_target = source_target.cuda()
            source_domain = torch.ones(source_input.size(0), dtype=torch.long)
            source_domain = source_domain.cuda()
            general_rep_soruce = general_model(source_input)
            domain_rep_source, logits_source_domain = domain_model(
                source_input)
            source_domain_loss = criterion(logits_source_domain, source_domain)

            data_target = data_target_iter.next()
            target_input, target_target = data_target
            target_input = target_input.cuda()
            target_target = target_target.cuda()
            target_domain = torch.zeros(target_input.size(0), dtype=torch.long)
            target_domain = target_domain.cuda()
            n = source_input.size(0)
            print(source_target, target_target)
            general_rep_target = general_model(target_input)
            domain_rep_target, logits_target_domain = domain_model(
                target_input)
            target_domain_loss = criterion(logits_target_domain, target_domain)

            total_domain_loss = source_domain_loss + target_domain_loss

            class_rep_source = general_rep_soruce - domain_rep_source
            class_rep_target = general_rep_target - domain_rep_target
            logits_class_source = classification_head(class_rep_source)
            logits_class_target = classification_head(class_rep_target)

            source_class_loss = criterion(
                logits_class_source, source_target)
            target_class_loss = criterion(
                logits_class_target, target_target)
            total_class_loss = source_class_loss+target_class_loss

            '''res['probs'] += [logits_class_source]
            res['probs'] += [logits_class_target]
            res['gt'] += [source_target]
            res['gt'] += [target_target]'''

            '''pred_s = logits_class_source.data.max(1, keepdim=True)[1]
            n_correct_source += pred_s.eq(
                source_target.data.view_as(pred_s)).sum().item()
            n_total += args.batch_size
            pred_t = logits_class_target.data.max(1, keepdim=True)[1]
            n_correct_target += pred_t.eq(
                target_target.data.view_as(pred_t)).sum().item()'''

            source_prec1, source_prec5 = utils.accuracy(
                logits_class_source, source_target, topk=(1, 5))
            target_prec1, target_prec5 = utils.accuracy(
                logits_class_target, target_target, topk=(1, 5))

            objs.update(total_domain_loss.item(), n)
            objs1.update(total_class_loss.item(), n)
            source_top1.update(source_prec1.item(), n)
            source_top5.update(source_prec5.item(), n)
            target_top1.update(target_prec1.item(), n)
            target_top5.update(target_prec5.item(), n)

            count += 1

    '''probs = torch.cat(res['probs'], dim=0)
    gt = torch.cat(res['gt'], dim=0)
    eval_res = utils.mean_accuracy(probs, gt)
    acc_source = n_correct_source * 1.0 / n_total
    acc_target = n_correct_target * 1.0 / n_total
    print(eval_res, acc_source, acc_target)
    writer.add_scalar("Domain loss/train", total_domain_loss, epoch)
    writer.add_scalar("Class loss/train", total_class_loss, epoch)
    writer.add_scalar("Acc/train/source", acc_source, epoch)
    writer.add_scalar("Acc/train/target", acc_target, epoch)
    writer.add_scalar("Mean Acc", eval_res, epoch)'''

    writer.add_scalar("Domain loss/valid", objs.avg, epoch)
    writer.add_scalar("Class loss/valid", objs1.avg, epoch)
    writer.add_scalar("Top1/valid/source", source_top1.avg, epoch)
    writer.add_scalar("Top5/valid/source", source_top5.avg, epoch)
    writer.add_scalar("Top1/valid/target", target_top1.avg, epoch)
    writer.add_scalar("Top5/valid/target", target_top5.avg, epoch)

    return source_top1.avg, target_top1.avg, objs1.avg


if __name__ == '__main__':
    main()
