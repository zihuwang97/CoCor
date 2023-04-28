import builtins
import torch.distributed as dist
import os
import torchvision.models as models
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import datetime
import time
from torchvision.datasets import ImageFolder

from model.cocor import CoCor
from model.cocor_siam import CoCor_siam
from ops.os_operation import mkdir
from data_processing.Multi_FixTransform import Multi_Fixtransform
from training.train_utils import adjust_learning_rate,adjust_lr,adj_lr_with_warmup,save_checkpoint
from training.train import train

class ImageNet100(ImageFolder):
    def __init__(self, root, transform):
        with open('data_processing/imagenet100.txt') as f:
            classes = [line.strip() for line in f]
            class_to_idx = { cls: idx for idx, cls in enumerate(classes) }

        super().__init__(os.path.join(root), transform=transform)
        samples = []
        for path, label in self.samples:
            cls = self.classes[label]
            if cls not in class_to_idx:
                continue
            label = class_to_idx[cls]
            samples.append((path, label))

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in samples]

def init_log_path(args):
    """
    :param args:
    :return:
    save model+log path
    """
    save_path = os.path.join(os.getcwd(), args.log_path)
    mkdir(save_path)
    save_path = os.path.join(save_path, args.dataset)
    mkdir(save_path)
    save_path = os.path.join(save_path, "Alpha_" + str(args.alpha))
    mkdir(save_path)
    save_path = os.path.join(save_path, "Aug_" + str(args.aug_times))
    mkdir(save_path)
    save_path = os.path.join(save_path, "lr_" + str(args.lr))
    mkdir(save_path)
    save_path = os.path.join(save_path, "cos_" + str(args.cos))
    mkdir(save_path)
    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')
    now = time.strftime("%H:%M:%S")
    save_path = os.path.join(save_path, formatted_today + now)
    mkdir(save_path)
    return save_path


def main_worker(gpu, ngpus_per_node, args):
    """
    :param gpu: current gpu id
    :param ngpus_per_node: number of gpus in one node
    :param args: config parameter
    :return:
    init training setup and iteratively training
    """
    params = vars(args)
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    print("=> creating model '{}'".format(args.arch))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    #init model
    if args.framework == 'moco':
        model = CoCor(models.__dict__[args.arch], args,
                                        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    elif args.framework == 'simsiam':
        model = CoCor_siam(args)
        if args.distributed: 
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        print("your framework %s is not supported for pretraining now" % args.framework)
        exit()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer_encoder = torch.optim.SGD(model.module.encoder_q.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer_d = torch.optim.SGD([{'params':model.module.mapping.parameters()}], args.d_lr,
                                momentum=args.momentum)

    optimizer_lin = torch.optim.SGD(model.module.projector.parameters(), args.lin_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # PMNN initialization
    pmnn_init_state = torch.load('./resume_state/monot_nn_init.pt', map_location='cuda:{}'.format(args.gpu))
    model.module.mapping.load_state_dict(pmnn_init_state)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            ###### load encoder_k, encoder_q, queue & queue_ptr ######
            # model_dict = model.state_dict()
            # pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            # model.load_state_dict(pretrained_dict)
            ###### -------------------------------------------- ######
            optimizer_encoder.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit()

    cudnn.benchmark = True
    # config data loader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    fix_transform = Multi_Fixtransform(args.size_crops,
                                       args.nmb_crops,
                                       args.min_scale_crops,
                                       args.max_scale_crops, normalize, args.aug_times)

    transform_train_lin = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])

    traindir = os.path.join(args.data, 'train')
    
    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     fix_transform)
    # train_dataset_lin = datasets.ImageFolder(
    #     traindir,
    #     transform_train_lin)
    train_dataset = ImageNet100(traindir, fix_transform)
    train_dataset_lin = ImageNet100(traindir, transform_train_lin)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_sampler_lin = torch.utils.data.distributed.DistributedSampler(train_dataset_lin)
        
    else:
        train_sampler = None
        train_sampler_lin = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    train_loader_lin = torch.utils.data.DataLoader(
        train_dataset_lin, batch_size=args.batch_size, shuffle=(train_sampler_lin is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler_lin)
    

    save_path=init_log_path(args) #config model save path and log path
    log_path = os.path.join(save_path,"train.log")
    best_Acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer_encoder, epoch, args)
        adjust_lr(optimizer_lin, optimizer_d, epoch, args)
        adj_lr_with_warmup(optimizer_d, epoch, wait=3, warmup=2, args=args)
        
        acc1 = train(train_loader, train_loader_lin, model, criterion, 
                      optimizer_encoder, optimizer_d, optimizer_lin, epoch, args,log_path)
        is_best = best_Acc > acc1
        best_Acc = max(best_Acc, acc1)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_dict = {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'best_acc': best_Acc,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer_encoder.state_dict(),
                }

            if epoch % 10 == 9:
                tmp_save_path = os.path.join(save_path, 'checkpoint_{:04d}.pth.tar'.format(epoch))
                save_checkpoint(save_dict, is_best=False, filename=tmp_save_path)
            tmp_save_path = os.path.join(save_path, 'checkpoint_best.pth.tar')
            save_checkpoint(save_dict, is_best=is_best, filename=tmp_save_path)

