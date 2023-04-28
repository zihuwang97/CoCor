import time
import torch.nn as nn
import torch

from training.train_utils import AverageMeter,ProgressMeter,accuracy

def train(train_loader, train_loader_lin, model, criterion_siam, criterion_lin, 
          optimizer_encoder, optimizer_d, optimizer_lin, epoch, args,log_path):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    train_iter_lin = iter(train_loader_lin)
    train_epoch_lin = 0
    for i, (images, _) in enumerate(train_loader):
        # switch to train mode
        model.train()
        # measure data loading time
        data_time.update(time.time() - end)

        try:
            images_l_t, labels_t = train_iter_lin.next()
        except:
            if args.world_size > 1:
                train_epoch_lin += 1
                train_loader_lin.sampler.set_epoch(train_epoch_lin)
            train_iter_lin = iter(train_loader_lin)
            images_l_t, labels_t = train_iter_lin.next()

        if args.gpu is not None:
            # two different augmented views
            for k in range(2):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            # resize-cropped views (no other augs applied)
            for k in range(len(images)-5,len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image_k = images[0]
            image_q = images[1]
            # parse diversely augmented views to get [a list of imgs] & [a list of aug compositions]
            st_trans_list = []
            image_strong_list = []
            for j in range(2,len(images)-5):
                image_strong = images[j][0].cuda(args.gpu, non_blocking=True)
                st_trans = images[j][1]
                st_trans = torch.stack(st_trans).T.cuda(args.gpu, non_blocking=True)
                src = torch.ones(image_k.size(0),14).cuda(args.gpu)
                st_trans = torch.zeros_like(src).cuda(args.gpu).scatter_add_(1,st_trans,src)
                st_trans_list.append(st_trans)
                image_strong_list.append(image_strong)
            image_cluster = images[-5:]
            images_l_t = images_l_t.cuda(args.gpu, non_blocking=True)
            labels_t = labels_t.cuda(args.gpu, non_blocking=True)

        p1, p2, z1, z2, q_strong_angle, d = model(image_q, image_k, image_strong_list, image_cluster, st_trans_list)
        pred = model.module.lin_forward(images_l_t)
        loss_l = criterion_lin(pred, labels_t)
        loss_contrastive = 0
        loss_angle = 0
       
        # contrastive loss
        loss_contrastive = -(criterion_siam(p1, z2).mean() + criterion_siam(p2, z1).mean()) * 0.5
        # consistency loss
        for k in range(len(q_strong_angle)):
            loss_angle += torch.nn.functional.softplus(d[k]-q_strong_angle[k])
        
        loss = loss_contrastive + args.alpha * loss_angle
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output[0], target[0], topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))
 
        # compute gradient and do SGD step
        optimizer_encoder.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_encoder.step()
        
        p1, p2, z1, z2, q_strong_angle_, d = model(image_q, image_k, image_strong_list, image_cluster, st_trans_list)
        loss_angle = 0

        # contrastive loss
        loss_contrastive = -(criterion_siam(p1, z2).mean() + criterion_siam(p2, z1).mean()) * 0.5
        # consistency loss
        for k in range(len(q_strong_angle_)):
            loss_angle += torch.nn.functional.softplus(d[k]-q_strong_angle_[k])
        loss_prime = loss_contrastive + args.alpha * loss_angle

        pred_ = model.module.lin_forward(images_l_t)
        loss_l_prime = criterion_lin(pred_, labels_t)

        # softmax factor update
        denom = loss_prime - loss
        denom = torch.clamp(denom,min=1e-4) if denom>=0 else torch.clamp(denom,max=-1e-4)
        diff = (loss_l_prime - loss_l)/denom
        loss_curve = 0
        for k in range(len(q_strong_angle_)):
            t = torch.exp(d[k]-q_strong_angle_[k]) / torch.square(1+torch.exp(d[k]-q_strong_angle_[k]))
            loss_curve += t.detach() * diff.detach() * (q_strong_angle_[k]-q_strong_angle[k]).detach() * d[k]
        loss_curve = loss_curve + 0*loss_prime #+ torch.sum(pred_)
        optimizer_d.zero_grad()
        loss_curve.backward()
        optimizer_d.step()

        # projector update
        optimizer_lin.zero_grad()
        loss_l_prime.backward()
        optimizer_lin.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            progress.write_record(i,log_path)
    return top1.avg

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    mAP = AverageMeter("mAP", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5, mAP],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            output = model.module.lin_forward(images)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1 = torch.mean(concat_all_gather(acc1.unsqueeze(0)), dim=0, keepdim=True)
            acc5 = torch.mean(concat_all_gather(acc5.unsqueeze(0)), dim=0, keepdim=True)
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            loss = criterion(output, target)
            losses.update(loss.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % 15 == 0:
            #     progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} mAP {mAP.avg:.3f} '
              .format(top1=top1, top5=top5, mAP=mAP))

    return top1.avg

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output