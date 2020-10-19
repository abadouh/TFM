import time

import torch
import torch.nn as nn
import math
#import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
#import torchvision.datasets as datasets
#import torchvision.models as models
from pytorch_model import resnet50
#from torchsummary import summary

from yaml_classification_dataset import YAMLClassificationDataset

best_acc1 = 0


""""
Helper functions

"""


def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = m.parameters()
    if only_trainable:
        parameters = list(p for p in parameters if p.requires_grad)
    unique = dict((p.data_ptr(), p) for p in parameters).values()
    return sum(p.numel() for p in unique)


def count_parameters(model):
    #table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        #table.add_row([name, param])
        total_params += param
    print(f"Total Trainable Params: {total_params}")
    return total_params


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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch % 30 == 0:
        lr = lr * (0.1 ** (epoch/30))

        for param_group in optimizer.param_groups:
            print("epoch: {}, old lr: {}, new_lr: {}".format(epoch, param_group['lr'], lr), flush=True)
            param_group['lr'] = lr


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_exec(train_loader, model, criterion, optimizer, epoch, gpu=False, gpuid=None, print_freq=10):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #print("This is {} iteration, this is what: {}".format(i, what))

        if gpuid is not None:
            images = images.cuda(gpuid, non_blocking=True)
            target = target.cuda(gpuid, non_blocking=True)
        elif gpu:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % print_freq == 0:
        progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1


def pytorch_env():

    print("Pytorch detected {} GPUs in the system".format(torch.cuda.device_count()))


"""
Main functions to be use from outside the module
"""


def pytorch_load_data_cifar(path='/home/abadouh/PycharmProjects/ML_playground/cifar/', batch_size=128,
                            load_subset=False, subset_size=4096):
    # Data loading code
    #    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    train_set = torchvision.datasets.CIFAR10(root=path, train=True, download=False, transform=transform)

    if load_subset:
        train_set = torch.utils.data.Subset(train_set,  range(0, subset_size))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)

    val_set = torchvision.datasets.CIFAR10(root=path, train=False, download=False, transform=transform)

    if load_subset:
        val_set = torch.utils.data.Subset(val_set, range(0, math.ceil(subset_size/4)))

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, val_loader, classes


def pytorch_load_data_isic(path='/home/abadouh/workspaces/thesis/isic_skin_lesion/', batch_size=128):
    # Data loading code
    #    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])
    validation_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_set = YAMLClassificationDataset(dataset=path, transform=train_transform, split=['training'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)

    val_set = YAMLClassificationDataset(dataset=path, transform=validation_transforms, split=['validation'])
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7')
    return train_loader, val_loader, classes


def pytorch_load_data_imagenet(path='/home/abadouh/workspaces/thesis/isic_skin_lesion/', batch_size=128):
    # Data loading code
    #    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])
    validation_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_set = YAMLClassificationDataset(dataset=path, transform=train_transform, split=['training'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)

    val_set = YAMLClassificationDataset(dataset=path, transform=validation_transforms, split=['validation'])
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    classes = []
    return train_loader, val_loader, classes


def pytorch_resnet50(gpu, gpuid, learning_rate=1e-2, momentum=0.9, weight_decay=1e-4, debug=False, dataset="cifar10"):

    ngpus_per_node = torch.cuda.device_count()

    if gpuid is not None:
        print("Use GPU: {} for training".format(gpuid))

    # create model
    print("=> creating model resnet50")

    # res_model = models.__dict__['resnet50']()

    if gpu:
        torch.cuda.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
    else:
        torch.manual_seed(1234)
        #torch.manual_seed_all(1234)

    res_model = resnet50(pretrained=False, progress=True, dataset=dataset)

    #if dataset.casefold() == "cifar10":
    #    res_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    #    res_model.fc = nn.Linear(2048, 10)
    #elif dataset.casefold() == "isic":
    #    res_model.fc = nn.Linear(2048, 8)
    #elif dataset.casefold() == "imagenet":
    #    print("Pytorch: no need to update the last FC layer in Imagenet")
    #    # res_model.fc = nn.Linear(2048, )
    #else:
    #    print("Please specify a dataset name, for more info run all_reasnet50.py -h")

    #model = nn.Sequential(*list(res_model.children())[0:3], *list(res_model.children())[4:])
    model = res_model

    if debug:
        # count_parameters(model)
        print("Total number of parameters:{}".format(numel(model)), flush=True)
        print("Total number of trainable-parameters:{}".format(numel(model, True)), flush=True)
        print(model)
        # summary(model, (3,32,32))

    if gpuid is not None:
        torch.cuda.set_device(gpuid)
        model = model.cuda(gpuid)
    elif gpu and ngpus_per_node > 0:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    # define loss function (criterion) and optimizer
    if gpuid is not None:
        criterion = nn.CrossEntropyLoss().cuda(gpuid)
        cudnn.benchmark = True
    elif gpu and ngpus_per_node > 0:
        criterion = nn.CrossEntropyLoss().cuda()
        cudnn.benchmark = True
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)

    return model, criterion, optimizer


def pytorch_validate(val_loader, model, criterion, gpu=False, gpuid=None, print_freq=10):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if gpuid is not None:
                images = images.cuda(gpuid, non_blocking=True)
                target = target.cuda(gpuid, non_blocking=True)
            elif gpu:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

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

            #if i % print_freq == 0:
            progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5), flush=True)

    return top1.avg


def pytorch_train(model, criterion, optimizer, train_loader, val_loader, learning_rate=1e-2,
                  gpu=False, gpuid=None, epochs=10, dynamic_lr=False, time2acc=False):

    global best_acc1

    train_time = AverageMeter('Time', ':6.3f')
    print("pytorch_train", flush=True)
    end = time.time()
    for epoch in range(epochs):
        if dynamic_lr and epoch > 0:
            adjust_learning_rate(optimizer, epoch, learning_rate)

        print("pytorch_train epoch: {} ".format(epoch), flush=True)

        # train for one epoch
        acc1_train = train_exec(train_loader, model, criterion, optimizer, epoch, gpu, gpuid)

        # evaluate on validation set
        acc1 = pytorch_validate(val_loader, model, criterion, gpu, gpuid)

        # remember best acc@1 and save checkpoint
        # is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if time2acc and acc1_train.avg >= 74.9:
            train_time.update(time.time() - end)
            print("The training set achieved 74.90% accuracy after {} epochs and total of {} ".format(epoch, train_time), flush=True)
            return
    train_time.update(time.time() - end)
    print("The training set achieved {}% accuracy after {} epochs and total of {} minuets".format(acc1_train,
                                                                                                  epochs, train_time), flush=True)
    return


"""
if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        and args.rank % ngpus_per_node == 0):
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer' : optimizer.state_dict(),
    }, is_best)
"""
