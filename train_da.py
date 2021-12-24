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
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import random
from torch.autograd import Variable
from model import NetworkCIFAR as Network
from resnet import *
from classification_head import *
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

parser = argparse.ArgumentParser("o31")
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='0,1', help='gpu device id')
parser.add_argument('--epochs', type=int, default=1000,
                    help='num of training epochs')
parser.add_argument('--init_channels', type=int,
                    default=32, help='num of init channels')
parser.add_argument('--layers', type=int, default=20,
                    help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true',
                    default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float,
                    default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true',
                    default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int,
                    default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float,
                    default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS',
                    help='which architecture to use')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float,
                    default=0.7, help='portion of training data')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

DOMAIN_CLASSES = 2
NUM_CLASSES = 31
#os.environ["NCCL_DEBUG"] = "INFO"


def main():
    np.random.seed(args.seed)
    # torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    logging.info('gpu devices = %d' % torch.cuda.device_count())

    print(torch.cuda.is_available())
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    t = torch.cuda.get_device_properties(0).total_memory
    print("==============================================================")
    print(t)
    print("==============================================================")
    logging.info('gpu devices = %d' % torch.cuda.device_count())
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    print(args.arch)
    genotype = eval("genotypes.%s" % args.arch)
    print(genotype)
    output_dim = args.init_channels*16
    domain_model = Network(
        args.init_channels, DOMAIN_CLASSES, args.layers, criterion, genotype)
    general_model = ResNet(criterion, output_dim)
    classification_head = ClassificationHead(output_dim, NUM_CLASSES)
    classification_head = classification_head.cuda()

    print("GPUS: ", torch.cuda.device_count())
    domain_model = nn.DataParallel(domain_model)
    domain_model = domain_model.cuda()
    general_model = nn.DataParallel(general_model)
    general_model = general_model.cuda()
    classification_head = nn.DataParallel(classification_head)
    classification_head = classification_head.cuda()

    print("TRAINING INITIALIZED")
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

    print("D-A")
    source_data_train = dset.ImageFolder(
        '../newdata/dslr/images/train', transform=train_transform)
    target_data_train = dset.ImageFolder(
        '../newdata/amazon/images/train', transform=train_transform)
    target_data_test = dset.ImageFolder(
        '../newdata/amazon/images/test', transform=valid_transform)
    print("source_data_train: ", len(source_data_train),
          " target_data_train: ", len(target_data_train),
          " target_data_test", len(target_data_test),)

    source_train_queue = torch.utils.data.DataLoader(source_data_train, batch_size=args.batch_size,
                                                     shuffle=True,
                                                     pin_memory=True,
                                                     num_workers=0)
    target_train_queue = torch.utils.data.DataLoader(target_data_train, batch_size=args.batch_size,
                                                     shuffle=True,
                                                     pin_memory=True,
                                                     num_workers=0)
    target_test_queue = torch.utils.data.DataLoader(target_data_test, batch_size=args.batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=0)

    print("source_train_queue: ", len(source_train_queue))
    print("target_train_queue: ", len(target_train_queue),
          " target_test_queue: ", len(target_test_queue))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs))
    best_train_acc_source = 0.0
    best_train_acc_target = 0.0
    best_eval_res = 0.0
    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        domain_model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        # training
        train_acc_source, train_acc_target, train_obj = train(source_train_queue, target_train_queue, domain_model,
                                                              general_model, classification_head, criterion, optimizer, optimizer_general, optimizer_class_head, epoch)
        logging.info('train_acc %f %f', train_acc_source, train_acc_target)
        # writer.flush()

        # validation
        eval_res1, eval_res2, valid_obj = infer(
            target_test_queue, domain_model, general_model, classification_head, criterion, epoch)
        logging.info('test_acc %f %f', eval_res1, eval_res2)
        # writer.flush()

        # writer.close()

        if train_acc_source > best_train_acc_source:
            best_train_acc_source = train_acc_source
        logging.info('train_acc_source %f, best_train_acc_source %f',
                     train_acc_source, best_train_acc_source)

        if train_acc_target > best_train_acc_target:
            best_train_acc_target = train_acc_target
        logging.info('train_acc_target %f, best_train_acc_target %f',
                     train_acc_target, best_train_acc_target)

        if eval_res1 > best_eval_res:
            best_eval_res = eval_res1
        logging.info('eval_res %f, best_eval_res %f',
                     eval_res1, best_eval_res)

        utils.save(domain_model, os.path.join(args.save, 'domain_weights.pt'))
        utils.save(general_model, os.path.join(
            args.save, 'general_weights.pt'))
        utils.save(classification_head, os.path.join(
            args.save, 'classification_head.pt'))


def train(source_train_queue, target_train_queue, domain_model,
          general_model, classification_head, criterion, optimizer, optimizer_general, optimizer_class_head, epoch):

    objs = utils.AvgrageMeter()
    objs1 = utils.AvgrageMeter()
    source_top1 = utils.AvgrageMeter()
    source_top5 = utils.AvgrageMeter()
    target_top1 = utils.AvgrageMeter()
    target_top5 = utils.AvgrageMeter()

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
        print("Input shape: ", source_input.shape)
        # print("Target shape: ", source_target.shape)
        # print("Domain shape: ", source_domain.shape)
        n = source_input.size(0)
        source_input = source_input.cuda()
        source_target = source_target.cuda(non_blocking=True)
        source_domain = torch.ones(source_input.size(0), dtype=torch.long)
        source_domain = source_domain.cuda(non_blocking=True)
        general_rep_soruce = general_model(source_input)
        _, _, logits_source_domain, logits_source_domain_aux = domain_model(
            source_input)
        source_domain_loss = criterion(logits_source_domain, source_domain)
        data_target = data_target_iter.next()
        target_input, target_target = data_target
        target_input = target_input.cuda()
        target_target = target_target.cuda(non_blocking=True)
        print(source_target, target_target)
        target_domain = torch.zeros(target_input.size(0), dtype=torch.long)
        target_domain = target_domain.cuda(non_blocking=True)
        general_rep_target = general_model(target_input)
        _, _, logits_target_domain, logits_target_domain_aux = domain_model(
            target_input)
        target_domain_loss = criterion(logits_target_domain, target_domain)
        if args.auxiliary:
            source_loss_aux = criterion(
                logits_source_domain_aux, source_domain)
            source_domain_loss += args.auxiliary_weight*source_loss_aux
            target_loss_aux = criterion(
                logits_target_domain_aux, target_domain)
            target_domain_loss += args.auxiliary_weight*target_loss_aux

        total_domain_loss = source_domain_loss + target_domain_loss

        optimizer.zero_grad()
        total_domain_loss.backward()
        nn.utils.clip_grad_norm(domain_model.parameters(), args.grad_clip)
        optimizer.step()

        domain_rep_source, domain_rep_source_aux, _, _ = domain_model(
            source_input)
        domain_rep_target, domain_rep_target_aux, _, _ = domain_model(
            target_input)

        class_rep_source = general_rep_soruce - domain_rep_source
        class_rep_target = general_rep_target - domain_rep_target
        logits_class_source = classification_head(class_rep_source)
        logits_class_target = classification_head(class_rep_target)
        source_class_loss = criterion(logits_class_source, source_target)
        target_class_loss = criterion(logits_class_target, target_target)

        if args.auxiliary:
            class_rep_source_aux = general_rep_soruce - domain_rep_source_aux
            class_rep_target_aux = general_rep_target - domain_rep_target_aux
            logits_class_source_aux = classification_head(class_rep_source_aux)
            logits_class_target_aux = classification_head(class_rep_target_aux)
            source_class_loss_aux = criterion(
                logits_class_source_aux, source_target)
            target_class_loss_aux = criterion(
                logits_class_target_aux, target_target)
            source_class_loss += args.auxiliary_weight*source_class_loss_aux
            target_class_loss += args.auxiliary_weight*target_class_loss_aux

        total_class_loss = source_class_loss+target_class_loss
        optimizer_general.zero_grad()
        optimizer_class_head.zero_grad()
        total_class_loss.backward()
        optimizer_general.step()
        optimizer_class_head.step()

        '''pred_s = logits_class_source.data.max(1, keepdim=True)[1]
        n_correct_source += pred_s.eq(
            source_target.data.view_as(pred_s)).sum().item()
        n_total += args.batch_size
        pred_t = logits_class_target.data.max(1, keepdim=True)[1]
        n_correct_target += pred_t.eq(
            target_target.data.view_as(pred_t)).sum().item()

        if count % args.report_freq == 0:
            logging.info('train %03d %e %e', count, source_class_loss,
                         source_class_loss)'''

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
        count += 1
    #acc_source = n_correct_source * 1.0 / n_total
    #acc_target = n_correct_target * 1.0 / n_total

    '''writer.add_scalars("Train/Loss/train/", {'domain loss': objs.avg,
                                             'class loss': objs1.avg
                                             }, epoch)
    writer.add_scalars("Train/Top1/train/", {'source': source_top1.avg,
                                             "target": target_top1.avg}, epoch)
    writer.add_scalars("Train/Top5/train/", {'source': source_top5.avg,
                                             "target": target_top5.avg}, epoch)'''

    return source_top1.avg, target_top1.avg, total_class_loss


def infer(target_test_queue, domain_model, general_model, classification_head, criterion, epoch):
    objs = utils.AvgrageMeter()
    objs1 = utils.AvgrageMeter()
    test_top1 = utils.AvgrageMeter()
    test_top5 = utils.AvgrageMeter()
    len_dataloader = len(target_test_queue)
    data_test_iter = iter(target_test_queue)
    count = 0

    domain_model.eval()
    with torch.no_grad():
        while count < len_dataloader:
            data_target = data_test_iter.next()
            target_input, target_target = data_target
            target_input = target_input.cuda()
            target_target = target_target.cuda()
            target_domain = torch.zeros(target_input.size(0), dtype=torch.long)
            target_domain = target_domain.cuda()
            general_rep_target = general_model(target_input)
            domain_rep_target, _, logits_target_domain, _ = domain_model(
                target_input)
            target_domain_loss = criterion(logits_target_domain, target_domain)

            class_rep_target = general_rep_target - domain_rep_target
            logits_class_target = classification_head(class_rep_target)

            target_class_loss = criterion(
                logits_class_target, target_target)
            print(target_target)

            target_prec1, target_prec5 = utils.accuracy(
                logits_class_target, target_target, topk=(1, 5))

            n = target_input.size(0)

            objs.update(target_domain_loss.item(), n)
            objs1.update(target_class_loss.item(), n)
            test_top1.update(target_prec1.item(), n)
            test_top5.update(target_prec5.item(), n)

            count += 1

    '''writer.add_scalars("Train/Loss/test", {'domain loss': objs.avg,
                                           'class loss': objs1.avg
                                           }, epoch)
    writer.add_scalars("Train/Acc/test/", {'Top1': test_top1.avg,
                                           "Top5": test_top5.avg}, epoch)'''

    return test_top1.avg, test_top5.avg, target_class_loss


if __name__ == '__main__':
    main()
