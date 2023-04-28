import torch
import torch.nn as nn

class monot_nn(nn.Module):
    def __init__(self, augpool_size: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.rand(augpool_size, 64))
        self.b1 = nn.Parameter(torch.rand(1, 64))
        self.w2 = nn.Parameter(torch.rand(64, 1))
        self.b2 = nn.Parameter(torch.rand(1))
        self.activate = nn.ReLU()

    def forward(self, x):
        w1 = torch.nn.functional.softplus(self.w1)
        w2 = torch.nn.functional.softplus(self.w2)
        d = x @ w1 + self.b1
        d = self.activate(d)
        d = d @ w2 + self.b2
        d = self.activate(d)
        # d = torch.sigmoid((d-5)/2)
        return 10-d

class CoCor(nn.Module):

    def __init__(self, base_encoder, args, dim=128, K=65536, m=0.999, T=0.2, mlp=True):
        """
        :param base_encoder: encoder model
        :param args: config parameters
        :param dim: feature dimension (default: 128)
        :param K: queue size; number of negative keys (default: 65536)
        :param m: momentum of updating key encoder (default: 0.999)
        :param T: softmax temperature (default: 0.2)
        :param mlp: use MLP layer to process encoder output or not (default: True)
        """
        super(CoCor, self).__init__()
        self.args = args
        self.K = K
        self.m = m
        self.T = T
        self.T2 = self.args.clsa_t

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        
        self.projector = base_encoder(num_classes=1000).fc

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        self.register_buffer("queue", torch.randn(dim, K))
        # self.register_buffer("queue_y", torch.randint(1000, (K,)))
        self.queue = nn.functional.normalize(self.queue, dim=0)  # normalize across queue instead of each example
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # config parameters for CLSA stronger augmentation and multi-crop
        self.weak_pick = args.pick_weak
        self.strong_pick = args.pick_strong
        self.weak_pick = set(self.weak_pick)
        self.strong_pick = set(self.strong_pick)
        self.gpu = args.gpu
        self.sym = self.args.sym

        self.mapping = monot_nn(14)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, queue, queue_ptr, keys):
        # gather keys before updating queue
        #keys = concat_all_gather(keys) #already concatenated before

        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def lin_forward(self, x):
        with torch.no_grad():
            x = self.encoder_q.conv1(x)
            x = self.encoder_q.bn1(x)
            x = self.encoder_q.relu(x)
            x = self.encoder_q.maxpool(x)

            x = self.encoder_q.layer1(x)
            x = self.encoder_q.layer2(x)
            x = self.encoder_q.layer3(x)
            x = self.encoder_q.layer4(x)

            x = self.encoder_q.avgpool(x)
            x = torch.flatten(x, 1)
        output = self.projector(x)

        return output

    def center_angle(self, x, centers):
        scores = x * centers
        scores = torch.sum(scores)
        return scores/x.size(0)

    def forward(self, im_q_list, im_k, im_strong_list, cluster_list, st_trans):
        """
        :param im_q_list: query image list
        :param im_k: key image
        :param im_strong_list: query strong image list
        :return:
        weak: logit_list, label_list
        strong: logit_list, label_list
        """
        q_list = []
        
        for i, im_q in enumerate(im_q_list):  # weak forward
            if i not in self.weak_pick:
                continue
            # can't shuffle because it will stop gradient only can be applied for k
            # im_q, idx_unshuffle = self._batch_shuffle_ddp(im_q)
            q = self.encoder_q(im_q) #+ randnoise # queries: NxC
            q = nn.functional.normalize(q, dim=1)
            # q = self._batch_unshuffle_ddp(q, idx_unshuffle)
            q_list.append(q)
    
        # compute key features
        with torch.no_grad():  # no gradient to keys
            # if update_key_encoder:
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k = k.detach()
            k = concat_all_gather(k)

        # compute logits
        # Einstein sum is more intuitive

        logits0_list = []
        labels0_list = []
        
        for choose_idx in range(len(q_list)):
            q = q_list[choose_idx]

            # positive logits: Nx1
            l_pos = torch.einsum('nc,ck->nk', [q, k.T])
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            cur_batch_size = logits.shape[0]
            cur_gpu = self.gpu
            choose_match = cur_gpu * cur_batch_size
            labels = torch.arange(choose_match, choose_match + cur_batch_size, dtype=torch.long).cuda()

            logits0_list.append(logits)
            labels0_list.append(labels)
    
        self._dequeue_and_enqueue(self.queue, self.queue_ptr, k)

        # centers=[]
        # cluster_list = [torch.cat(cluster_list[0:5],0), torch.cat(cluster_list[5:10],0), torch.cat(cluster_list[10:15],0)]
        # with torch.no_grad():
        #     for i in range(3):
        #         rep = self.encoder_q(cluster_list[i])
        #         rep = nn.functional.normalize(rep,dim=1)
        #         center = rep[0:32,...]+rep[32:64,...]+rep[64:96,...]+rep[96:128,...]+rep[128:160,...]
        #         center = nn.functional.normalize(center,dim=1)
        #         centers.append(center)
        centers=[]
        cluster_list = [torch.cat(cluster_list[0:5],0)]
        with torch.no_grad():
            for i in range(1):
                rep = self.encoder_q(cluster_list[i])
                rep = nn.functional.normalize(rep,dim=1)
                center = rep[0:64,...]+rep[64:128,...]+rep[128:192,...]+rep[192:256,...]+rep[256:320,...]
                center = nn.functional.normalize(center,dim=1)
                centers.append(center)

        # compute cluster centers
        q_strong_angle = []
        for i, im_strong in enumerate(im_strong_list):
            q_strong = self.encoder_q(im_strong)  # queries: NxC
            q_strong = nn.functional.normalize(q_strong, dim=1)
            if i<=1:
                angles = self.center_angle(x=q_strong, centers=centers[0].detach())
            elif i<=3:
                angles = self.center_angle(x=q_strong, centers=centers[1].detach())
            else:
                angles = self.center_angle(x=q_strong, centers=centers[2].detach())
            q_strong_angle.append(angles)

        d = []
        for trans in st_trans:
            d.append(torch.mean(self.mapping(trans)))
    
        return logits0_list, labels0_list, q_strong_angle, d


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

def cluster_centers(data, labels):
    clusters, count = torch.unique(labels, sorted=True, return_counts=True)
    idx_in = torch.isin(torch.arange(1000).cuda(),clusters)
    count_all = torch.ones(1000,dtype=torch.long).cuda()
    count_all[idx_in] = count
    M = torch.zeros(1000, len(data)).to(data.device)
    M[labels, torch.arange(len(data))] = 1
    M = torch.nn.functional.normalize(M, p=1, dim=1)
    sum = torch.mm(M, data)
    centers = torch.div(sum.t(), count_all).t()
    centers = torch.nn.functional.normalize(centers, dim=1)
    return centers