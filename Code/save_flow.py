import nibabel as nib
import argparse

import torch
from torch.utils.data import DataLoader

from monai.data import CSVDataset
from monai.networks.nets import BasicUNet
from monai.transforms import Compose, LoadImageD, ToTensord,  EnsureChannelFirstD,  ResizeD 

from tqdm import tqdm

from utils import transform


import time


basic_1 = Compose([
    LoadImageD(keys=["immA", "immB", "immGT"]),
    EnsureChannelFirstD(keys=["immA", "immB", "immGT"], channel_dim = 'no_channel'),
    # ResizeD(keys=["immA", "immB", "immGT"], spatial_size=(121, 145, 113)),
    ResizeD(keys=["immA", "immB", "immGT"], spatial_size=(182, 218, 170)),
    ToTensord(keys=["immA", "immB", "immGT"]),

])

transform = Compose([
    basic_1,
    transform.preprocessing(['immA', 'immB']),
])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="../Data/test.csv")
    
    args = parser.parse_args()

    
    test_dataset = CSVDataset(src= args.dataset, transform=transform, skiprows=1) 
    
    train_loader = DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=False)
    

    with open(args.dataset, 'r') as file:
        test_set = file.readlines()

   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    UNet = BasicUNet(spatial_dims=3, in_channels= 2, out_channels = 1, features=(32, 32, 64, 128, 256, 32)).to(device)
    # UNet.load_state_dict(torch.load("../Model/Unet.pth"))
    UNet.load_state_dict(torch.load("/storage/data_4T/riccardoraciti/unet/training_senza_resize/b4_0_4k_max/Unet-388.pth")) 


    for i in tqdm(range(1, len(test_set))):
        test_dataset = transform({"immA": test_set[i].split(',')[0].replace("\n", ""), 
                               "immB" : test_set[i].split(',')[1].replace("\n", ""),
                               "immGT": test_set[i].split(',')[-1].replace("\n", "")})
        
        target = UNet(test_dataset["images"].unsqueeze(0).to(device))
        mri = nib.nifti1.Nifti1Image(((target.squeeze(0).squeeze(0).cpu().detach().numpy() - 2000) / 1000), None) 
        # mri.to_filename(f"../Results/{test_set[i].split(',')[0].split('/')[5]}_flow.nii.gz")
        mri.to_filename(f"/storage/data_4T/riccardoraciti/dataset/Flow_noResize/{test_set[i].split(',')[0].split('/')[5]}_flow.nii.gz")


       