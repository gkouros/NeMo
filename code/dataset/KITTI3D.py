from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torch
import BboxTools as bbt


class KITTI3D(Dataset):
    def __init__(self, transforms, enable_cache=True, **kwargs):
        self.root_path = kwargs['rootpath']
        self.img_class = kwargs['imgclass']
        if 'data_pendix' in kwargs.keys():
            data_pendix = kwargs['data_pendix']
        else:
            data_pendix = ''

        if 'for_test' in kwargs:
            self.for_test = kwargs['for_test']
        else:
            self.for_test = False

        if 'anno_path' in kwargs:
            anno_path = kwargs['anno_path']
        else:
            anno_path = 'annotations'

        if 'img_path' in kwargs:
            img_path = kwargs['img_path']
        else:
            img_path = 'images'

        if 'list_path' in kwargs:
            list_path = kwargs['list_path']
        else:
            list_path = 'lists'

        self.image_path = os.path.join(self.root_path, img_path, '%s/' % (self.img_class))
        self.annotation_path = os.path.join(self.root_path, anno_path, '%s/' % (self.img_class))
        list_path = os.path.join(self.root_path, list_path, '%s/' % (self.img_class + data_pendix))

        self.transforms = transforms
        self.subtypes = kwargs['subtypes']

        self.file_list = sum(
            [[l.strip() for l in open(os.path.join(list_path, subtype_ + '.txt')).readlines()] for subtype_ in
             self.subtypes], [])

        if 'weighted' in kwargs:
            self.weighted = kwargs['weighted']
        else:
            self.weighted = False

        self.enable_cache = enable_cache
        self.cache_img = dict()
        self.cache_anno = dict()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        # get next img name
        name_img = self.file_list[item]

        # get img if in cache else load it from file
        if name_img in self.cache_anno.keys():
            annotation_file = self.cache_anno[name_img]
            img = self.cache_img[name_img]
        else:
            img = Image.open(os.path.join(self.image_path, name_img))

            # tackle the gray images.
            if img.mode != 'RGB':
                img = img.convert('RGB')
            annotation_file = np.load(os.path.join(self.annotation_path, name_img.split('.')[0] + '.npz'),
                                      allow_pickle=True)

            # if cache is enabled store img in dictionary
            if self.enable_cache:
                self.cache_anno[name_img] = dict(annotation_file)
                self.cache_img[name_img] = img

        # get bounding box annotation
        box_obj = bbt.from_numpy(annotation_file['box_obj'])
        # get mask from bounding box
        obj_mask = np.zeros(box_obj.boundary, dtype=np.float32)
        box_obj.assign(obj_mask, 1)

        # get keypoints list annotation
        kp = annotation_file['cropped_kp_list']
        # get visibility flag of keypoints
        iskpvisible = annotation_file['visible'] == 1 # 1-> visible, 2->occluded, 0->cropped?

        # weight keypoints if flag is set and there are weights in the annotation
        if self.weighted:
            iskpvisible = iskpvisible * annotation_file['kp_weights']

        # if not for testing but for training?
        if not self.for_test:
            # if all points are visible set visible flag
            iskpvisible = np.logical_and(iskpvisible, np.all(kp >= np.zeros_like(kp), axis=1))
            # ensure all keypoints are within image space
            iskpvisible = np.logical_and(iskpvisible, np.all(kp < np.array([img.size[::-1]]), axis=1))

        # clip keypoints to [0,0]-[h,w]
        kp = np.max([np.zeros_like(kp), kp], axis=0)
        kp = np.min([np.ones_like(kp) * (np.array([img.size[::-1]]) - 1), kp], axis=0)

        # extract image name from filename
        this_name = name_img.split('.')[0]

        # arrange pose into an array [distance, elevation, azimuth, theta]
        pose_ = np.array([5, annotation_file['elevation'], annotation_file['azimuth'], annotation_file['theta']], dtype=np.float32)

        # arrange data into a dictionary
        sample = {'img': img, 'kp': kp, 'iskpvisible': iskpvisible, 'this_name': this_name, 'obj_mask': obj_mask,
                  'box_obj': box_obj.shape, 'pose': pose_}

        # apply transformations (hflip, norm, tensor, color jitter)
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def get_image_size(self):
        name_img = self.file_list[0]
        img = Image.open(os.path.join(self.image_path, name_img))
        return np.array(img).shape[0:2]


class ToTensor(object):
    def __init__(self):
        self.trans = transforms.ToTensor()

    def __call__(self, sample):
        sample['img'] = self.trans(sample['img'])
        if not type(sample['iskpvisible']) == torch.Tensor:
            sample['iskpvisible'] = torch.Tensor(sample['iskpvisible'])
        if not type(sample['kp']) == torch.Tensor:
            sample['kp'] = torch.Tensor(sample['kp'])
        return sample


class Normalize(object):
    def __init__(self):
        self.trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        sample['img'] = self.trans(sample['img'])
        return sample


def hflip(sample):
    sample['img'] = transforms.functional.hflip(sample['img'])
    sample['kp'][:, 1] = sample['img'].size[0] - sample['kp'][:, 1] - 1
    return sample


class RandomHorizontalFlip(object):
    def __init__(self):
        self.trans = transforms.RandomApply([lambda x: hflip(x)], p=0.5)

    def __call__(self, sample):
        sample = self.trans(sample)

        return sample


class ColorJitter(object):
    def __init__(self):
        self.trans = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.4, hue=0)

    def __call__(self, sample):
        sample['img'] = self.trans(sample['img'])

        return sample
