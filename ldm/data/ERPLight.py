import os
import numpy as np
import PIL
from PIL import Image
import imageio as im
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import torch

import matplotlib.pyplot as plt

"""
    To Do:
        - add masking

    Notes:
        - data_root must be full path
"""

def h_rotate(img, angle):
    rot = int((angle/360) * img.shape[1])
    print(f"Shape: {img.shape}, Rot:{rot}, Angle: {angle}")
    return np.concatenate((img[:,int(rot):,:],img[:,:int(rot),:]),axis=1)

def h_rotate_torch(img, angle):
    #_, _, W = img.shape
    rot = int((angle/360) * img.shape[2])
    return torch.cat((img[...,rot:],img[...,:rot]),dim=-1)

def gamma_correct(img, gamma=2.2, alpha=1, inverse=0, percentile=50):
    clamped = np.where(img<0,0,img)
    if inverse:
        return np.power((alpha*clamped),gamma)
    else:
        gam_img = np.power(clamped,(1/gamma))
        if alpha == 0:
            alpha = 2*np.percentile(gam_img,percentile)
        return (1/alpha)*gam_img

class ERPLightBase(Dataset):
    def __init__(self,
                 csv_file,
                 data_root="/mnt/fast/nobackup/users/jh00695/UTransformer/UTransformer-main/dataset",
                 ldr_dir=None,
                 hdr_dir=None,
                 env_map_dir=None,
                 size=256,
                 interpolation="bicubic",
                 flip_p=0.5,
                 data_key="LDR",
                 flip_IO = False,
                 gamma = 6.6,
                 alpha = 1
                 ):
        self.data_root = data_root
        # with open(self.data_paths, "r") as f:
        #     self.image_paths = f.read().splitlines()
        df = pd.read_csv(os.path.join(self.data_root,csv_file))
        cols = df.columns
        self.image_paths = df['filename']
        if 'rotation' in cols:
            self.rot_list = df['rotation']
            self.flip_list = df['flip']
        else:
            self.rot_list = np.zeros_like(self.image_paths)
            self.flip_list = np.zeros_like(self.image_paths)

        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "LDR_file_path_": [os.path.join(self.data_root, ldr_dir, l+".png")
                           for l in self.image_paths],
            "HDR_file_path_": [os.path.join(self.data_root, hdr_dir, l+".exr")
                           for l in self.image_paths],
            "EM_file_path_": [os.path.join(self.data_root, env_map_dir, l+".exr")
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        data_keys = ["LDR","HDR","Env_Map","ALL"]
        if data_key in data_keys:
            self.data_key = data_key
        else:
            self.data_key = "LDR"
        self.IO = flip_IO
        self.gamma = gamma
        self.alpha = alpha

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        if "LDR" in self.data_key or "All" in self.data_key:
            image = Image.open(example["LDR_file_path_"])
            if not image.mode == "RGB":
                image = image.convert("RGB")

            # default to score-sde preprocessing
            # img = np.array(image).astype(np.uint8)
            # crop = min(img.shape[0], img.shape[1])
            # h, w, = img.shape[0], img.shape[1]
            # img = img[(h - crop) // 2:(h + crop) // 2,
            #       (w - crop) // 2:(w + crop) // 2]

            #image = Image.fromarray(img)
            if self.size is not None:
                image = image.resize((int(self.size*2),self.size), resample=self.interpolation)

            image = self.flip(image)
            image = np.array(image).astype(np.uint8)
            if self.rot_list[i]:
                image = h_rotate(image,self.rot_list[i])
            if self.IO:
                img = np.concatenate((img[:,self.size:,:],img[:,:self.size,:]),axis=1)
            #example["image"] = (image / 127.5 - 1.0).astype(np.float32)
            example["image"] = torch.as_tensor(image / 127.5 - 1.0).permute(2,0,1).cuda()

        if "HDR" in self.data_key or "All" in self.data_key:
            img = im.imread(example["HDR_file_path_"], 'EXR-FI')
            img = cv2.resize(img[...,:3], (int(self.size*2),self.size), interpolation=cv2.INTER_CUBIC)
            if self.flip_list[index]:
                img = cv2.flip(img,1)
            if self.rot_list[i]:
                img = h_rotate(img,self.rot_list[i])
            if self.IO:
                img = np.concatenate((img[:,self.size:,:],img[:,:self.size,:]),axis=1)
            example["HDR_image"] = gamma_correct(img, gamma=self.gamma, alpha=self.alpha) - 1

        if "Env_Map" in self.data_key or "All" in self.data_key:
            img = im.imread(example["EM_file_path_"], 'EXR-FI')
            img = cv2.resize(img[...,:3], (int(self.size*2),self.size), interpolation=cv2.INTER_CUBIC)
            if self.flip_list[index]:
                img = cv2.flip(img,1)
            if self.rot_list[i]:
                img = h_rotate(img,self.rot_list[i])
            if self.IO:
                img = np.concatenate((img[:,self.size:,:],img[:,:self.size,:]),axis=1)
            example["EM_image"] = gamma_correct(img, gamma=self.gamma, alpha=self.alpha) - 1

        return example


class ERPLightTrain(ERPLightBase):
    def __init__(self, **kwargs):
        super().__init__(csv_file="inout_aug4_set_train.csv", **kwargs)


class ERPLightValidation(ERPLightBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(csv_file="inout_aug4_set_val.csv", flip_p=flip_p, **kwargs)

class ERPLightTest(ERPLightBase):
    def __init__(self, **kwargs):
        super().__init__(csv_file="../data/test_csv.csv", **kwargs)


class LSUNBedroomsTrain(ERPLightBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_train.txt", data_root="data/lsun/bedrooms", **kwargs)


class LSUNBedroomsValidation(ERPLightBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_val.txt", data_root="data/lsun/bedrooms",
                         flip_p=flip_p, **kwargs)


class LSUNCatsTrain(ERPLightBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/cat_train.txt", data_root="data/lsun/cats", **kwargs)


class LSUNCatsValidation(ERPLightBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/cat_val.txt", data_root="data/lsun/cats",
                         flip_p=flip_p, **kwargs)

if __name__ == "__main__":
    dataset = ERPLightTest(data_root="../../assets/",data_key="LDR",size=256,ldr_dir="./",hdr_dir="./",env_map_dir="./")
    for images in dataset:
        img = images["image"]
        plt.imshow(img/2+0.5)
        plt.show()


