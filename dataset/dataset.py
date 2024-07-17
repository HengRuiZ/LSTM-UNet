import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import random

import sys
sys.path.append('../../CRNN')

from utils import dice_coeff




class CRNNSingleOrganDataset(Dataset):
    def __init__(self, data_dir1, data_dir2, patient, organ, img_size, interpolate, fractions):
        # data_dir should include subdirectories: CT, CBCT, CT Structures, CBCT Structures
        # CT is treated as the first fraction.
        data_path1 = Path(data_dir1)
        data_path2 = Path(data_dir2)

        self.image_list=[]
        self.mask_list=[]
        self.fractions=[]

        # add CT
        self.image_list.append(str(data_path1 / 'CT' / (patient + '.npy')))
        self.mask_list.append(str(data_path1 / 'CT Structures' / patient / organ))
        self.fractions.append('CT')

        # add CBCT and DCBCT
        dcbct_folder = data_path2 / 'DCBCT' / patient
        dcbcts_folder = data_path2 / 'DCBCT Structures' / patient
        #add interpolated CBCT
        for fr in fractions[:interpolate]:
            self.fractions.append(fr)
            self.image_list.append(str(dcbct_folder / fr / (patient + '.npy')))
            self.mask_list.append(str(dcbcts_folder/fr/organ))
        #add CBCT
        self.fractions.append('CBCT')
        self.image_list.append(str(data_path1 / 'CBCT' / (patient + '.npy')))
        self.mask_list.append(str(data_path1 / 'CBCT Structures' / 'cascaded_VTN' / patient / organ))
        #add extrapolated CBCT
        for fr in fractions[interpolate:]:
            self.fractions.append(fr)
            self.image_list.append(str(dcbct_folder / fr / (patient + '.npy')))
            self.mask_list.append(str(dcbcts_folder/fr/organ))
        self.interpolate = interpolate
        self.image_size = img_size
        # organ structure
        self.outChannel = 1
        # CBCT, last CBCT, last CBCTS
        self.inChannel = 3
        return
    def __len__(self):
        # CBCT, DCBCT
        return len(self.fractions)-1
    def resize(self,img,size):
        img = cv2.resize(img,dsize=size, interpolation=cv2.INTER_CUBIC)
        img = np.array(img)
        return img
    def load_image(self,image_path,img_size):
        img=np.load(image_path).astype(np.float32)
        img=self.resize(img,img_size)
        img=img/1000
        return img[:,:,2:90]
    def load_mask(self,mask_path,img_size):
        mask = np.load(mask_path).astype(np.float32)
        mask = self.resize(mask, img_size)
        mask = np.clip(mask, a_min=0, a_max=1).round()
        if mask.shape[2]==93:
            mask=mask[:,:,2:90]
        return mask
    def __getitem__(self, fr):
        # get fr_nd fraction data of the patient
        cur_cbct=self.load_image(self.image_list[fr+1],self.image_size)
        cur_cbcts=self.load_mask(self.mask_list[fr+1],self.image_size)
        last_cbct=self.load_image(self.image_list[fr],self.image_size)
        last_cbcts=self.load_mask(self.mask_list[fr],self.image_size)

        image = np.stack([cur_cbct,last_cbct,last_cbcts],axis=0)
        mask_input = np.expand_dims(last_cbcts,axis=0)
        mask = np.expand_dims(cur_cbcts,axis=0)
        image=np.expand_dims(image,axis=0)
        mask_input=np.expand_dims(mask_input,axis=0)
        mask=np.expand_dims(mask,axis=0)
        return {
            'image': torch.as_tensor(image.copy()).contiguous(),
            'mask_input': torch.as_tensor(mask_input.copy()).contiguous(),
            'mask': torch.as_tensor(mask.copy()).contiguous()
        }


class CRNNDirectSegDataset(CRNNSingleOrganDataset):
    def __init__(self, data_dir1, data_dir2, patient, organ, img_size, interpolate, fractions):
        # data_dir should include subdirectories: CT, CBCT, CT Structures, CBCT Structures
        # CT is treated as the first fraction.
        super(CRNNDirectSegDataset).__init__(data_dir1, data_dir2, patient, organ, img_size, interpolate, fractions)
        # organ structure
        self.outChannel = 1
        # CBCT, last CBCT, last CBCTS
        self.inChannel = 1
        return
    def __getitem__(self, fr):
        # get fr_nd fraction data of the patient
        cur_cbct = self.load_image(self.image_list[fr + 1], self.image_size)
        cur_cbcts = self.load_mask(self.mask_list[fr + 1], self.image_size)

        image = np.expand_dims(cur_cbct, axis=0)
        mask = np.expand_dims(cur_cbcts, axis=0)
        mask_input=np.zeros_like(mask)
        return {
            'image': torch.as_tensor(image.copy()).contiguous(),
            'mask_input': torch.as_tensor(mask_input.copy()).contiguous(),
            'mask': torch.as_tensor(mask.copy()).contiguous()
        }

class CRNNDropoutDataset(CRNNSingleOrganDataset):
    # randomly dropout the prior information
    def __init__(self, data_dir1, data_dir2, patient, organ, img_size, interpolate, fractions):
        # data_dir should include subdirectories: CT, CBCT, CT Structures, CBCT Structures
        # CT is treated as the first fraction.
        super(CRNNDropoutDataset).__init__(data_dir1, data_dir2, patient, organ, img_size, interpolate, fractions)
        return

    def __getitem__(self, fr):
        # get fr_nd fraction data of the patient
        cur_cbct = self.load_image(self.image_list[fr + 1], self.image_size)
        cur_cbcts = self.load_mask(self.mask_list[fr + 1], self.image_size)
        last_cbct = self.load_image(self.image_list[fr], self.image_size)
        last_cbcts = self.load_mask(self.mask_list[fr], self.image_size)

        image = np.stack([cur_cbct, last_cbct, last_cbcts], axis=0)
        if random.randint(0,1):
            image[1:]=0
        mask_input = np.expand_dims(last_cbcts, axis=0)
        mask = np.expand_dims(cur_cbcts, axis=0)
        image = np.expand_dims(image, axis=0)
        mask_input = np.expand_dims(mask_input, axis=0)
        mask = np.expand_dims(mask, axis=0)
        return {
            'image': torch.as_tensor(image.copy()).contiguous(),
            'mask_input': torch.as_tensor(mask_input.copy()).contiguous(),
            'mask': torch.as_tensor(mask.copy()).contiguous()
        }

class EthosSingleOrganDataset(Dataset):
    def __init__(self, data_dir, organ, img_size):
        # data_dir should include subdirectories: CT, CBCT, CT Structures, CBCT Structures
        # CT is treated as the first fraction.
        data_path = Path(data_dir)

        self.image_list=[]
        self.mask_list=[]
        self.fractions=listdir(data_dir)

        # add sCT and organs
        for fr in self.fractions:
            self.image_list.append(str(data_path / fr / 'sCT.npy'))
            self.mask_list.append(str(data_path / fr / organ))

        self.image_size = img_size
        # organ structure
        self.outChannel = 1
        # CBCT, last CBCT, last CBCTS
        self.inChannel = 3
        return
    def __len__(self):
        # CBCT, DCBCT
        return len(self.fractions)-1
    def resize(self,img,size):
        img = cv2.resize(img,dsize=size, interpolation=cv2.INTER_CUBIC)
        img = np.array(img)
        return img
    def load_image(self,image_path,img_size):
        img=np.load(image_path).astype(np.float32)
        img=self.resize(img,img_size)
        img=img/1000
        return img
    def load_mask(self,mask_path,img_size):
        mask = np.load(mask_path).astype(np.float32)
        mask = self.resize(mask, img_size)
        mask = np.clip(mask, a_min=0, a_max=1).round()
        return mask
    def __getitem__(self, fr):
        # get fr_nd fraction data of the patient
        cur_cbct=self.load_image(self.image_list[fr+1],self.image_size)
        cur_cbcts=self.load_mask(self.mask_list[fr+1],self.image_size)
        last_cbct=self.load_image(self.image_list[fr],self.image_size)
        last_cbcts=self.load_mask(self.mask_list[fr],self.image_size)

        image = np.stack([cur_cbct,last_cbct,last_cbcts],axis=0)
        mask = np.expand_dims(cur_cbcts,axis=0)
        mask_input = np.expand_dims(last_cbcts,axis=0)
        image=np.expand_dims(image,axis=0)
        mask_input=np.expand_dims(mask_input,axis=0)
        mask=np.expand_dims(mask,axis=0)
        return {
            'image': torch.as_tensor(image.copy()).contiguous(),
            'mask': torch.as_tensor(mask.copy()).contiguous(),
            'mask_input':torch.as_tensor(mask_input.copy()).contiguous()
        }

class CRNNSingleOrganDatasets:
    def __init__(self, data_dir1, data_dir2, organ, img_size, interpolate, DS=0, fractions=('-10','-05','05','10')):
        data_path1 = Path(data_dir1)
        patient_names=[splitext(p)[0] for p in listdir(data_path1/'CT')]
        self.patient_names=[]
        self.organ=organ.split('.')[0]
        self.img_size=img_size
        self.interpolate=interpolate
        self.fractions=fractions
        for n in patient_names:
            if os.path.exists(str(data_path1/'CT Structures'/n/organ)) \
                    and os.path.exists(str(data_path1/'CBCT Structures'/'cascaded_VTN'/n/organ)):
                self.patient_names.append(n)
        self.patients_number=len(self.patient_names)
        if self.patients_number==0:
            raise RuntimeError(f'No input file found in {data_dir1}, make sure you put your images there')
        logging.info(f'Creating dataset with {self.patients_number} patients')
        self.datasets=[]
        self.dataloaders=[]
        for p in self.patient_names:
            if DS==1:
                d = CRNNDirectSegDataset(data_dir1, data_dir2, p, organ, img_size, interpolate,fractions)
            elif DS==2:
                d = CRNNDropoutDataset(data_dir1, data_dir2, p, organ, img_size, interpolate, fractions)
            elif DS==0:
                d = CRNNSingleOrganDataset(data_dir1, data_dir2, p, organ, img_size, interpolate, fractions)
            self.datasets.append(d)
            self.dataloaders.append(DataLoader(d, batch_size=1, shuffle=False,num_workers=1))
        self.train_set = self.datasets[:-10]
        self.val_set = self.datasets[-10:-5]
        self.test_set = self.datasets[-5:]
        logging.info(f'train_set:{self.patient_names[:-10]}')
        logging.info(f'val_set:{self.patient_names[-10:-5]}')
        logging.info(f'test_set:{self.patient_names[-5:]}')
        return


class EthosSingleOrganDatasets:
    def __init__(self, data_dir, organ, img_size):
        data_path = Path(data_dir)
        patient_names=listdir(data_path)
        self.patient_names=[]
        self.organ=organ.split('.')[0]
        self.img_size=img_size
        for n in patient_names:
            if os.path.exists(str(data_path/n/'FX01'/organ)):
                self.patient_names.append(n)
        self.patients_number=len(self.patient_names)
        if self.patients_number==0:
            raise RuntimeError(f'No input file found in {data_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {self.patients_number} patients')
        self.datasets=[]
        self.dataloaders=[]
        for p in self.patient_names:
            d = EthosSingleOrganDataset(data_dir+f'/{p}', organ, img_size)
            self.datasets.append(d)
            self.dataloaders.append(DataLoader(d, batch_size=1, shuffle=False,num_workers=1))
        self.train_set=self.datasets[:4]
        self.val_set=self.datasets[4:6]
        self.test_set=self.datasets[6:]
        logging.info(f'train_set:{self.patient_names[:4]}')
        logging.info(f'val_set:{self.patient_names[4:6]}')
        logging.info(f'test_set:{self.patient_names[6:]}')
        return


def images_dice(images):
    l=len(images)
    dices=np.zeros(shape=[l,l],dtype=np.float32)
    for i in range(l):
        for j in range(l):
            dices[i,j]=dice_coeff(torch.as_tensor(images[i]).contiguous(),
                                  torch.as_tensor(images[j]).contiguous())
    return dices

def get_dices(dataset):
    def load_masks(dir):
        im=np.load(dir)
        im=np.round(im.astype(np.float32)).clip(min=0,max=1)
        if im.shape[2]==93:
            im=im[:,:,2:90]
        return im
    images=[]
    for mask_dir in dataset.mask_list:
        images.append(load_masks(mask_dir))
    return images_dice(images)

if __name__ == '__main__':
    #data = EthosSingleOrganDataset(data_dir='D:\S438175\data\EthosDataNPY\EHN012',organ='brachialplex_l.npy',img_size=[256,256])
    #data = CRNNSingleOrganDatasets(data_dir1='D:\S438175\data\DL_DIR_combine_data',data_dir2='D:\S438175\data\DL_DIR_combine_data', organ='BrachialPlex_L.npy',img_size=[256, 256],interpolate=2)
    data = EthosSingleOrganDatasets(data_dir=r'D:\S438175\data\EthosDataNPY',organ='brachialplex_l.npy',img_size=[256,256])
    patient=data.datasets[0]
    for i in range(len(patient)):
        slice=patient.__getitem__(i)
        pass
    pass
