import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

import monai
from monai.data import CSVDataset
from monai.networks.nets import BasicUNet
from torch.optim import AdamW
from monai.transforms import SpacingD, LoadImage, Compose, MapTransform, LoadImageD, ToTensord, AddChannelD, EnsureChannelFirstD, CropForegroundd, ResizeWithPadOrCropD, ResizeD
from monai.config import KeysCollection

from tqdm import tqdm
import numpy as np

from utils import transform

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

basic_1 = Compose([
    LoadImageD(keys=["immA", "immB", "immGT"]),
    #EnsureChannelFirstD(keys=["immA", "immB", "immGT"], channel_dim = 'no_channel'),
    AddChannelD(keys=["immA", "immB", "immGT"]),
    ResizeD(keys=["immA", "immB", "immGT"], spatial_size=(121, 145, 113)),
    #CropForegroundd(keys=["immA", "immB", "immGT"], source_key="immA"),
    #SpacingD(keys=["immA", "immB", "immGT"], pixdim=1.5),
    #ResizeWithPadOrCropD(keys=["immA", "immB", "immGT"], spatial_size=(130, 130, 130), mode='minimum'),
    ToTensord(keys=["immA", "immB", "immGT"]),

])

transform = Compose([
    basic_1,
    transform.preprocessing(['immA', 'immB']),
])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, default = "/home/alessiarondinella/rraciti/Brain-Change-Estimation/Data/test.csv", help='Path csv of test dataset.')
    parser.add_argument('--dir_load_model', type=str, default="/storage/data_4T/alessiarondinella_data/Brain-Change-Estimation/result", help='Path directory where to load checkpoint.')
    parser.add_argument('--dir_save_results', type=str, default="/storage/data_4T/alessiarondinella_data/Brain-Change-Estimation/result/test", help='Path directory where to save results.')

    args = parser.parse_args()


    UNet = BasicUNet(spatial_dims=3, in_channels= 2, out_channels = 1, features=(32, 32, 64, 128, 256, 32)).to(device)
    UNet.load_state_dict(torch.load("/storage/data_4T/alessiarondinella_data/Brain-Change-Estimation/results_2/Unet-84.pth"))

    test_dataset = transform({"immA": "/storage/data_4T/lpuglisi-siena/SIENA/I96321_to_I727661/A_halfwayto_B_brain.nii.gz", 
                               "immB" : "/storage/data_4T/lpuglisi-siena/SIENA/I96321_to_I727661/B_halfwayto_A_brain.nii.gz", 
                               "immGT":"/storage/data_4T/lpuglisi-siena/SIENA/I96321_to_I727661/A_to_B_flow.nii.gz"})
    
    print(test_dataset)

    print(test_dataset.keys())

    print(test_dataset["images"].size())

    out=UNet(test_dataset["images"].unsqueeze(0).to(device))
    out= out.squeeze(0)

    import nibabel as nib
    mri = nib.nifti1.Nifti1Image(out.squeeze(0).cpu().detach().numpy(), test_dataset['immGT_meta_dict']['original_affine']) #np.eye(4)
    mri.to_filename('/storage/data_4T/alessiarondinella_data/Brain-Change-Estimation/results_2/result.nii.gz')

    mri = nib.nifti1.Nifti1Image(test_dataset["images"][0].squeeze(0).cpu().detach().numpy(), test_dataset['immGT_meta_dict']['original_affine']) #np.eye(4)
    mri.to_filename('/storage/data_4T/alessiarondinella_data/Brain-Change-Estimation/results_2/input1.nii.gz')

