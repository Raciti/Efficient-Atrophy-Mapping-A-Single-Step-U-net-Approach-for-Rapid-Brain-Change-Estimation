import argparse

import torch
from torch.utils.data import DataLoader

import monai
from monai.data import CSVDataset
from monai.networks.nets import BasicUNet
from monai.transforms import SpacingD, LoadImage, Compose, MapTransform, LoadImageD, ToTensord,  EnsureChannelFirstD, CropForegroundd, ResizeWithPadOrCropD, ResizeD #AddChannelD,

import numpy as np
import nibabel as nib

from transform import preprocessing as preprocessing

import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

basic_1 = Compose([
    LoadImageD(keys=["immA", "immB", "immGT"]),
    EnsureChannelFirstD(keys=["immA", "immB", "immGT"], channel_dim = 'no_channel'),
    #AddChannelD(keys=["immA", "immB", "immGT"]),
    ResizeD(keys=["immA", "immB", "immGT"], spatial_size=(121, 145, 113)),
    #CropForegroundd(keys=["immA", "immB", "immGT"], source_key="immA"),
    #SpacingD(keys=["immA", "immB", "immGT"], pixdim=1.5),
    #ResizeWithPadOrCropD(keys=["immA", "immB", "immGT"], spatial_size=(130, 130, 130), mode='minimum'),
    ToTensord(keys=["immA", "immB", "immGT"]),

])

transform = Compose([
    basic_1,
    preprocessing(['immA', 'immB']),
])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, default = "/home/alessiarondinella/rraciti/Brain-Change-Estimation/Data/test.csv", help='Path csv of test dataset.')
    parser.add_argument('--dir_load_model', type=str, default="/storage/data_4T/alessiarondinella_data/Brain-Change-Estimation/result", help='Path directory where to load checkpoint.')
    parser.add_argument('--dir_save_results', type=str, default="/storage/data_4T/alessiarondinella_data/Brain-Change-Estimation/result/test", help='Path directory where to save results.')

    args = parser.parse_args()

    train_dataset = CSVDataset(src="Data/val.csv", transform=transform, skiprows=1) 
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=1,
                            shuffle=True)
    
    min = 100
    max = 0
    i = 0
    for batch in train_loader:
        print(i)
        GT = nib.nifti1.Nifti1Image(((batch["immGT"].cpu().detach().numpy() - 2000) / 1000),None)
        if min >  GT.get_fdata().min(): 
            min = GT.get_fdata().min()
        if max < GT.get_fdata().max():
            max = GT.get_fdata().max()

        i += 1
    print(f"Val GT max: {max} \nVAL GT min: {min}\n")

# Train GT max: 2.3681235313415527 
# Train GT min: -2.6399033069610596
# Val GT max: 2.4687159061431885 
# VAL GT min: -2.359320640563965
