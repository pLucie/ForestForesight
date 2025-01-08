# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from logging import getLogger

import numpy as np
from torch.utils.data import Dataset
import torch
#import kornia.augmentation as K
#import torch.nn as nn

logger = getLogger()


def randomCrop(img, mask, width, height):
    assert img.shape[1] == mask.shape[0]
    assert img.shape[2] == mask.shape[1]

    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[2] - height)
    img = img[:,x:x+width,y:y+height]
    mask = mask[x:x+width,y:y+height]
    return img, mask


class DataloaderEval(Dataset):
    def __init__(self, 
                 img, 
                 coords, 
                 psize = 128,
                 labels=None,
                 lab_ind=True):

        super(DataloaderEval, self).__init__()
        self.img = img
        self.coord = coords
        self.lab_ind = lab_ind
        self.labels = labels
        self.psize = psize
        
        
    def __len__(self):
        return self.coord.shape[0]
        
    def __getitem__(self, idx):
        
        image = self.img[:,self.coord[idx,0]-self.psize//2:self.coord[idx,0]+self.psize//2+self.psize%2,
                              self.coord[idx,1]-self.psize//2:self.coord[idx,1]+self.psize//2+self.psize%2]

        image = torch.from_numpy(image.astype(np.float32))
        
        if self.lab_ind:
            mask_img = self.labels[self.coord[idx,0]-self.psize//2:self.coord[idx,0]+self.psize//2+self.psize%2,
                                  self.coord[idx,1]-self.psize//2:self.coord[idx,1]+self.psize//2+self.psize%2]
            mask_img = torch.from_numpy(mask_img.astype(np.float32))
            return image, mask_img
        
        return image




class DatasetFromCoord(Dataset):
    def __init__(self, 
                image = None,
                labels = None,
                coords = None, 
                psize = None,
                samples=False,
                evaluation = False,
                augment = False):

        super(DatasetFromCoord, self).__init__()
        self.image = image
        self.labels = labels
        self.coord = coords
        self.psize = psize
        self.samples = samples
        self.evaluation = evaluation
        
        self.augment = augment
        
        if self.augment:
            self.trans = K.AugmentationSequential(K.RandomResizedCrop(size=(self.psize,self.psize),scale=(0.5, 1.), p=0.0),
                                              # K.RandomRotation(degrees=(-45.0,45.0), p=0.2),
                                              # K.RandomElasticTransform(kernel_size=(3, 3), sigma=(5.0, 5.0), alpha=(1.0, 1.0),p=0.2),
                                              K.RandomPosterize(3., p=0.2),
                                              K.RandomEqualize(p=0.2),
                                              K.RandomMotionBlur(3, 10., 0.5),
                                              # K.RandomSharpness(sharpness=0.5),
                                              K.RandomGaussianBlur((3, 3), (0.1, 2.0),p=0.5),
                                              # K.RandomPerspective(0.1, p=0.5),
                                              # K.RandomThinPlateSpline(p=0.5),
                                              data_keys=['input','mask'],
                                              keepdim = True,
                                              same_on_batch=False)
        
    def __len__(self):
        if self.samples:
            return self.samples
        else:
            return len(self.coord)
        
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.image[:,self.coord[idx,0]-self.psize//2:self.coord[idx,0]+self.psize//2,
                              self.coord[idx,1]-self.psize//2:self.coord[idx,1]+self.psize//2]
        
        if not self.evaluation:
            ref = self.labels[self.coord[idx,0]-self.psize//2:self.coord[idx,0]+self.psize//2,
                                   self.coord[idx,1]-self.psize//2:self.coord[idx,1]+self.psize//2]
             
            # image,ref = randomCrop(image,ref,64,64)
            
            ref = torch.tensor(ref.astype(np.float32)) 
            image = torch.from_numpy(image.astype(np.float32))
            
            if self.augment:
                image, ref = self.trans(image,ref)

            return image, ref.long()
        
        image = torch.from_numpy(image.astype(np.float32))
        return image
    
class Dataloader(Dataset):
    def __init__(self, 
                 img, 
                 coords, 
                 psize = 128,
                 samples=None,
                 labels=None,
                 lab_ind=True):

        super(Dataloader, self).__init__()
        self.img = img
        self.coord = coords
        self.samples = samples
        self.lab_ind = lab_ind
        self.labels = labels
        self.psize = psize
        
        
    def __len__(self):
        return self.samples
        
    def __getitem__(self, idx):
        time_indx = np.random.randint(0,self.img.shape[1])
        
        if self.coord[time_indx].shape[0] <= idx:
            idx = np.random.randint(0,self.coord[time_indx].shape[0])
        
        image = self.img[:,time_indx, self.coord[time_indx][idx,0]-self.psize//2:self.coord[time_indx][idx,0]+self.psize//2+self.psize%2,
                              self.coord[time_indx][idx,1]-self.psize//2:self.coord[time_indx][idx,1]+self.psize//2+self.psize%2]

        image = torch.from_numpy(image.astype(np.float32))
        
        if self.lab_ind:
            mask_img = self.labels[time_indx, self.coord[time_indx][idx,0]-self.psize//2:self.coord[time_indx][idx,0]+self.psize//2+self.psize%2,
                                  self.coord[time_indx][idx,1]-self.psize//2:self.coord[time_indx][idx,1]+self.psize//2+self.psize%2]
            mask_img = torch.from_numpy(mask_img.astype(np.float32))
            # lab_img = self.labels[:,self.coord[idx,0],self.coord[idx,1]]
            # lab_img = torch.from_numpy(lab_img.astype(np.float32))
            return image, mask_img
        
        return image
    
class DataloaderCNN(Dataset):
    def __init__(self, 
                 img, 
                 coords, 
                 psize = 128,
                 samples=None,
                 labels=None,
                 lab_ind=True):

        super(DataloaderCNN, self).__init__()
        self.img = img
        self.coord = coords
        self.samples = samples
        self.lab_ind = lab_ind
        self.labels = labels
        self.psize = psize
        
        
    def __len__(self):
        if self.samples:
            return self.samples
        else:
            return self.coord.shape[0]
        
    def __getitem__(self, idx):
        image = self.img[:,self.coord[idx,0]-self.psize//2:self.coord[idx,0]+self.psize//2+self.psize%2,
                              self.coord[idx,1]-self.psize//2:self.coord[idx,1]+self.psize//2+self.psize%2]

        image = torch.from_numpy(image.astype(np.float32))
        
        if self.lab_ind:
            lab = self.labels[self.coord[idx,0],self.coord[idx,1]]
            lab = torch.from_numpy(np.array(lab).astype(np.float32))
            return image, lab
        
        return image
    
class DataloaderLSTM(Dataset):
    def __init__(self, 
                 img, 
                 coords, 
                 lstm_length = 6,
                 samples=None,
                 labels=None,
                 lab_ind=True):

        super(DataloaderLSTM, self).__init__()
        self.img = img
        self.coord = coords
        self.samples = samples
        self.lab_ind = lab_ind
        self.labels = labels
        self.lstm_length = lstm_length
        self.temp_size = self.img.shape[0]
        
        
    def __len__(self):
        if self.samples:
            return self.samples
        else:
            return self.coord.shape[0]
        
    def __getitem__(self, idx):
        if self.lab_ind:
            tmp_idx = np.random.randint(self.lstm_length,self.temp_size-1)
            image = self.img[tmp_idx-self.lstm_length:tmp_idx,:,self.coord[idx,0],self.coord[idx,1]]
            
        if not self.lab_ind:
            image = self.img[-self.lstm_length:,:,self.coord[idx,0],self.coord[idx,1]]

        image = torch.from_numpy(image.astype(np.float32))
        
        if self.lab_ind:
            lab = np.sum(self.labels[tmp_idx+1,self.coord[idx,0],self.coord[idx,1]])
            lab = torch.from_numpy(np.array(lab).astype(np.float32))
            return image, lab
        
        return image
    
    
class DataloaderMLP(Dataset):
    def __init__(self, 
                 img, 
                 coords, 
                 samples=None,
                 labels=None,
                 lab_ind=True):

        super(DataloaderCNN, self).__init__()
        self.img = img
        self.coord = coords
        self.samples = samples
        self.lab_ind = lab_ind
        self.labels = labels
        
        
    def __len__(self):
        if self.samples:
            return self.samples
        else:
            return self.coord.shape[0]
        
    def __getitem__(self, idx):
        pixel = self.img[:,self.coord[idx,0],self.coord[idx,1]]

        pixel = torch.from_numpy(pixel.astype(np.float32))
        
        if self.lab_ind:
            lab = self.labels[:,self.coord[idx,0],self.coord[idx,1]]
            lab = torch.from_numpy(lab.astype(np.float32))
            return pixel, lab
        
        return pixel
    
class MemmapNPYCropDataset(Dataset):
    def __init__(self, 
                 file_img, 
                 file_lab, 
                 coords, 
                 max_time,
                 psize,
                 tile_shape,
                 features,
                 samples):

        super(MemmapNPYCropDataset, self).__init__()
        self.file_img = file_img
        self.file_lab = file_lab
        self.coord = coords
        self.samples = samples
        self.psize = psize
        self.max_time = max_time
        self.tile_shape = tile_shape
        self.features = features
        
        
    def __len__(self):
        return self.samples
        
    def __getitem__(self, idx):
        file_indx = np.random.randint(0,len(self.file_img))
        time_indx = np.random.randint(0,self.max_time)
        
        # Get the filename and corresponding coordinate
        filename = self.file_img[file_indx]
        refname = self.file_lab[file_indx]
        coords = self.coord[file_indx][time_indx]
        if coords.shape[0] <= idx:
            idx = np.random.randint(0,len(self.file_img)) 

        row, col = coords[idx]
        # Define the region to load
        row_start = max(0, row - self.psize // 2)
        col_start = max(0, col - self.psize // 2)
        row_end = row_start + self.psize +self.psize%2
        col_end = col_start + self.psize + +self.psize%2

        # Open the file as a memory-mapped array
        img = np.memmap(filename, dtype=np.float16, mode='r', shape=(self.features, self.max_time, self.tile_shape, self.tile_shape))
        ref = np.memmap(refname, dtype=np.uint8, mode='r', shape=(self.max_time, self.tile_shape, self.tile_shape))
        crop_img = torch.from_numpy(img[:, time_indx, row_start:row_end, col_start:col_end].copy().astype(np.float32))
        crop_ref = torch.from_numpy(ref[time_indx, row_start:row_end, col_start:col_end].copy().astype(np.float32))

        return crop_img, crop_ref
    

class NumpyTileDataset(Dataset):
    def __init__(self, file_img, file_lab, coords, psize, samples, max_time):
        self.file_img = file_img
        self.file_lab = file_lab
        self.coords = coords
        self.tile_size = psize
        self.samples = samples
        self.max_time = max_time

    def __len__(self):
        if self.samples:
            return self.samples
        else:
            return len(self.coords[0])

    def __getitem__(self, idx):
        file_idx = np.random.randint(0,len(self.file_img))        
        time_indx = np.random.randint(0,self.max_time)
        
        coords = self.coords[file_idx][time_indx]
        
        if coords.shape[0] <= idx:
            idx = np.random.randint(0,len(self.file_img)) 

        x, y = coords[idx]
        
        file_img = self.file_img[file_idx]
        file_lab = self.file_lab[file_idx]
        
        # Load the numpy file as a memory-mapped array
        data = np.load(file_img, mmap_mode='r')
        ref = np.load(file_lab, mmap_mode='r')
        
        # Define the region to load
        row_start = max(0, x - self.tile_size // 2)
        col_start = max(0, y - self.tile_size // 2)
        row_end = row_start + self.tile_size +self.tile_size%2
        col_end = col_start + self.tile_size + +self.tile_size%2
        
        # Crop the tile
        tile = data[:, time_indx, row_start:row_end, col_start:col_end]
        gt = ref[time_indx, row_start:row_end, col_start:col_end]
        
        # Convert to tensor
        tile = torch.from_numpy(tile).float()
        gt = torch.from_numpy(gt.astype(np.float32)).float()
        
        return tile, gt
