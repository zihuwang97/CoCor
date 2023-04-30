import torch
import torch.nn as nn
from model.build_models import load_backbone, load_mlp

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

class CoCor_siam(nn.Module):

    def __init__(self, args, T=0.2):
        """
        :param base_encoder: encoder model
        :param args: config parameters
        :param dim: feature dimension (default: 128)
        :param K: queue size; number of negative keys (default: 65536)
        :param m: momentum of updating key encoder (default: 0.999)
        :param T: softmax temperature (default: 0.2)
        :param mlp: use MLP layer to process encoder output or not (default: True)
        """
        super(CoCor_siam, self).__init__()
        self.args = args
        self.T = T
        self.T2 = self.args.clsa_t

        # create the encoders
        # num_classes is the output fc dimension
        self.backbone, num_backbone_features = load_backbone(args.arch)
        self.projector = load_mlp(num_backbone_features,
                                n_hidden=2048,
                                n_out=2048,
                                num_layers=3,
                                last_bn=True)
        self.predictor = load_mlp(2048,
                                512,
                                2048,
                                num_layers=2,
                                last_bn=False)
        self.classifier = nn.Linear(num_backbone_features, 100)  # 100 classes for IMAGENET100

        # config parameters for CLSA stronger augmentation and multi-crop
        self.weak_pick = args.pick_weak
        self.strong_pick = args.pick_strong
        self.weak_pick = set(self.weak_pick)
        self.strong_pick = set(self.strong_pick)
        self.gpu = args.gpu

        self.mapping = monot_nn(14)

    def lin_forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        output = self.classifier(x)
        return output

    def center_angle(self, x, centers):
        scores = x * centers
        scores = torch.sum(scores)
        return scores/x.size(0)

    def forward(self, im_q, im_k, im_strong_list, cluster_list, st_trans):
        """
        :param im_q_list: query image list
        :param im_k: key image
        :param im_strong_list: query strong image list
        :return:
        weak: logit_list, label_list
        strong: logit_list, label_list
        """
        z1 = self.projector(self.backbone(im_k)) # NxC
        z2 = self.projector(self.backbone(im_q)) # NxC
        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC
    
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
    
        return p1, p2, z1.detach(), z2.detach(), q_strong_angle, d


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