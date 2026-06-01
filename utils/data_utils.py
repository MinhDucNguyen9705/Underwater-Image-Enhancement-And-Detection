import os
import random
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop

def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def random_crop(img, degrade_img, img_size=(256, 256)):
    
    th, tw = img_size
    w, h = TF.get_image_size(img)
    
    if h < th or w < tw:
        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)
        padding = [
            pad_w // 2, 
            pad_h // 2, 
            (pad_w + 1) // 2, 
            (pad_h + 1) // 2
        ]
        
        img = TF.pad(img, padding, padding_mode='reflect')
        degrade_img = TF.pad(degrade_img, padding, padding_mode='reflect')

    i, j, h_crop, w_crop = RandomCrop.get_params(img, output_size=img_size)
    
    return TF.crop(img, i, j, h_crop, w_crop), TF.crop(degrade_img, i, j, h_crop, w_crop)

def random_augmentation(img, degrade_img, flip_v=True, rot=True):

    if random.random() < 0.5 and flip_v:
        img = TF.vflip(img)
        degrade_img = TF.vflip(degrade_img)

    if rot:
        if random.random() < 0.5:
            img = TF.hflip(img)
            degrade_img = TF.hflip(degrade_img)
        if random.random() < 0.5:
            rot_ang = random.choice([90, 180, 270])
            img = TF.rotate(img, rot_ang)
            degrade_img = TF.rotate(degrade_img, rot_ang)
        
    return img, degrade_img

class UnderwaterDataset(Dataset):
    def __init__(self, degrade_dir, ref_dir, img_size=None, train=True):
        img_names = os.listdir(degrade_dir)
        self.degrade_paths = [os.path.join(degrade_dir, img_name) for img_name in img_names]
        self.ref_paths = [os.path.join(ref_dir, img_name) for img_name in img_names]
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.degrade_paths)

    def __getitem__(self, idx):

        degrade_path = self.degrade_paths[idx]
        ref_path = self.ref_paths[idx]

        degrade_img = load_image(degrade_path)
        ref_img = load_image(ref_path)

        if ref_img.shape != degrade_img.shape:
            ref_img = cv2.resize(ref_img, (degrade_img.shape[0], degrade_img.shape[1]))

        degrade_img = torch.from_numpy(degrade_img).permute(2, 0, 1) / 255.
        ref_img = torch.from_numpy(ref_img).permute(2, 0, 1) / 255.

        if self.train:
            ref_img, degrade_img = random_crop(ref_img, degrade_img, self.img_size)
            ref_img, degrade_img = random_augmentation(ref_img, degrade_img)
        else:
            if self.img_size is not None:
                degrade_img = TF.resize(degrade_img, size=self.img_size)
                ref_img = TF.resize(ref_img, size=self.img_size)

        return degrade_img, ref_img

class UnderwaterDatasetNonRef(Dataset):
    def __init__(self, degrade_dir, img_size=None, train=True):
        img_names = os.listdir(degrade_dir)
        self.degrade_paths = [os.path.join(degrade_dir, img_name) for img_name in img_names]
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.degrade_paths)

    def __getitem__(self, idx):

        degrade_path = self.degrade_paths[idx]
        degrade_img = load_image(degrade_path)
        degrade_img = torch.from_numpy(degrade_img).permute(2, 0, 1) / 255.
        img_shape = degrade_img.shape
        ref_img = torch.randn(img_shape)

        if self.train:
            ref_img, degrade_img = random_crop(ref_img, degrade_img, self.img_size)
            ref_img, degrade_img = random_augmentation(ref_img, degrade_img)
        else:
            if self.img_size is not None:
                degrade_img = TF.resize(degrade_img, size=self.img_size)
    
        return degrade_img