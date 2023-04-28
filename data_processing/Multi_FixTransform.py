#modified from https://github.com/facebookresearch/swav/blob/master/src/multicropdataset.py
from torchvision import transforms
from data_processing.RandAugment import RandAugment
from data_processing.Image_ops import GaussianBlur

import torch
import math
import warnings
from typing import List, Optional, Tuple, Union
from collections.abc import Sequence
from torch import Tensor
import torchvision.transforms.functional as F
# from torchvision import _log_api_usage_once
# from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode
# from torchvision.transforms import _setup_size

class Compose_idx(transforms.Compose):
    def __call__(self, img):
        idx = None
        for t in self.transforms:
            if isinstance(img, list):
                idx = img[1]
                img = img[0]
                img = t(img)
            else:
                img = t(img)
        if idx:
            return [img, idx]
        else:
            return img

class Multi_Fixtransform(object):
    def __init__(self,
            size_crops,
            nmb_crops,
            min_scale_crops,
            max_scale_crops,normalize,
            aug_times,init_size=224):
        """
        :param size_crops: list of crops with crop output img size
        :param nmb_crops: number of output cropped image
        :param min_scale_crops: minimum scale for corresponding crop
        :param max_scale_crops: maximum scale for corresponding crop
        :param normalize: normalize operation
        :param aug_times: strong augmentation times
        :param init_size: key image size
        """
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        trans=[]
        #key image transform
        self.weak = transforms.Compose([
            transforms.RandomResizedCrop(init_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        trans.append(self.weak)
        self.aug_times=aug_times
        trans_weak=[]
        trans_strong=[]
        trans_clsf=[]
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )

            strong = Compose_idx([
            randomresizedcrop,
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            RandAugment(n=self.aug_times, m=10),
            transforms.ToTensor(),
            normalize
            ])

            weak=transforms.Compose([
            randomresizedcrop,
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])

            trans_weak.extend([weak]*nmb_crops[i])
            trans_strong.extend([strong]*nmb_crops[i])

        clsf_trans1 = transforms.Compose([
        transforms.RandomResizedCrop(224, (0.2, 1.0)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
        ])
        
        # clsf_trans2 = transforms.Compose([
        # transforms.RandomResizedCrop(160, (0.143, 0.715)),
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        # ], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # normalize
        # ])

        # clsf_trans3 = transforms.Compose([
        # transforms.RandomResizedCrop(96, (0.086, 0.429)),
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        # ], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # normalize
        # ])

        trans_clsf.extend(5*[clsf_trans1])
        # trans_clsf.extend(5*[clsf_trans2])
        # trans_clsf.extend(5*[clsf_trans3])
        trans.extend(trans_weak)
        trans.extend(trans_strong)
        trans.extend(trans_clsf)
        self.trans=trans
    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.trans))
        return multi_crops


# class Multi_Fixtransform(object):
#     def __init__(self,
#             size_crops,
#             nmb_crops,
#             min_scale_crops,
#             max_scale_crops,normalize,
#             aug_times,init_size=224):
#         assert len(size_crops) == len(nmb_crops)
#         assert len(min_scale_crops) == len(nmb_crops)
#         assert len(max_scale_crops) == len(nmb_crops)
#         trans=[]
#         #key image transform
#         self.weak = transforms.Compose([
#             transforms.RandomResizedCrop(init_size, scale=(0.2, 1.)),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
#             transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize
#         ])
#         trans.append(self.weak)
#         self.aug_times=aug_times
#         trans_weak=[]
#         trans_strong=[]
#         trans_clsf=[]
#         for i in range(len(size_crops)):
#             randomresizedcrop = transforms.RandomResizedCrop(
#                 size_crops[i],
#                 scale=(min_scale_crops[i], max_scale_crops[i]),
#             )

#             strong = Compose_idx([
#             randomresizedcrop,
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
#             transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
#             transforms.RandomHorizontalFlip(),
#             RandAugment(n=self.aug_times, m=10),
#             transforms.ToTensor(),
#             normalize
#             ])

#             weak=transforms.Compose([
#             randomresizedcrop,
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
#             transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize
#             ])

#             trans_weak.extend([weak]*nmb_crops[i])
#             trans_strong.extend([strong]*nmb_crops[i])

#         clsf_trans1 = transforms.Compose([
#         transforms.RandomResizedCrop(224, (0.2, 1.0)),
#         transforms.RandomApply([
#             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#         ], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize
#         ])

#         trans_clsf.extend([clsf_trans1])
#         trans.extend(trans_weak)
#         # trans.extend(trans_strong)
#         trans.extend(trans_clsf)
#         self.trans=trans
#         self.trans_strong = trans_strong
#     def __call__(self, x):
#         '''
#         return image list:
#         1.
#         2.
#         3.
#         '''
#         multi_crops = list(map(lambda trans: trans(x), self.trans))
#         strong_auged = list(map(lambda trans: trans(x), self.trans_strong))
#         multi_crops.extend(strong_auged)
#         return multi_crops

# class DeterminedResizedCrop(transforms.RandomResizedCrop):
#     def __init__(
#         self,
#         size,
#         scale=(0.08, 1.0),
#         ratio=(3.0 / 4.0, 4.0 / 3.0),
#         interpolation=InterpolationMode.BILINEAR,
#         antialias: Optional[Union[str, bool]] = "warn",
#     ):
#         super().__init__()
#         _log_api_usage_once(self)
#         self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

#         if not isinstance(scale, Sequence):
#             raise TypeError("Scale should be a sequence")
#         if not isinstance(ratio, Sequence):
#             raise TypeError("Ratio should be a sequence")
#         if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
#             warnings.warn("Scale and ratio should be of kind (min, max)")

#         if isinstance(interpolation, int):
#             interpolation = _interpolation_modes_from_int(interpolation)

#         self.interpolation = interpolation
#         self.antialias = antialias
#         self.scale = scale
#         self.ratio = ratio

#     @staticmethod
#     def get_params(img: Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
#         _, height, width = F.get_dimensions(img)
#         area = height * width

#         log_ratio = torch.log(torch.tensor(ratio))
#         for _ in range(10):
#             target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
#             aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

#             w = int(round(math.sqrt(target_area * aspect_ratio)))
#             h = int(round(math.sqrt(target_area / aspect_ratio)))

#             if 0 < w <= width and 0 < h <= height:
#                 i = torch.randint(0, height - h + 1, size=(1,)).item()
#                 j = torch.randint(0, width - w + 1, size=(1,)).item()
#                 return i, j, h, w

#         # Fallback to central crop
#         in_ratio = float(width) / float(height)
#         if in_ratio < min(ratio):
#             w = width
#             h = int(round(w / min(ratio)))
#         elif in_ratio > max(ratio):
#             h = height
#             w = int(round(h * max(ratio)))
#         else:  # whole image
#             w = width
#             h = height
#         i = (height - h) // 2
#         j = (width - w) // 2
#         return i, j, h, w

#     def forward(self, i, j, h, w, img):
#         return F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)

