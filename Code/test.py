import argparse

import torch

from monai.networks.nets import BasicUNet
from monai.transforms import Compose, LoadImageD, ToTensord,  EnsureChannelFirstD, ResizeD 

from utils import losses

from utils import transform


import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

basic_1 = Compose([
    LoadImageD(keys=["immA", "immB", "immGT"]),
    EnsureChannelFirstD(keys=["immA", "immB", "immGT"], channel_dim = 'no_channel'),
    ResizeD(keys=["immA", "immB", "immGT"], spatial_size=(121, 145, 113)),
    ToTensord(keys=["immA", "immB", "immGT"]),

])

transform = Compose([
    basic_1,
    transform.preprocessing(['immA', 'immB']),
])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, default = "Data/test.csv", help='Path csv of test dataset.')
    parser.add_argument('--dir_load_model', type=str, default="Result", help='Path directory where to load checkpoint.')
    parser.add_argument('--dir_save_results', type=str, default="Result/test", help='Path directory where to save results.')

    args = parser.parse_args()


    UNet = BasicUNet(spatial_dims=3, in_channels= 2, out_channels = 1, features=(32, 32, 64, 128, 256, 32)).to(device)
    UNet.load_state_dict(torch.load("Model/Unet.pth"))

    test_dataset = transform({"immA": "Data/A_halfwayto_B_brain.nii.gz", 
                               "immB" : "Data/B_halfwayto_A_brain.nii.gz", 
                               "immGT":"Data/A_to_B_flow.nii.gz"})

    loss_function = losses.CustomMSELoss(2, "max")

    start_time = time.time()

    out=UNet(test_dataset["images"].unsqueeze(0).to(device))

    end_time = time.time()

    print(f"\nEsecution time: {round(end_time-start_time,2)}s.\n")

    loss = loss_function((out - 2000) / 1000, (test_dataset["immGT"].squeeze(0).to(device) - 2000) / 1000)
    print(f"Loss MSE: {loss.item()}.\n")

    import nibabel as nib
    #Save output
    mri = nib.nifti1.Nifti1Image(((out.cpu().detach().numpy() - 2000) / 1000), None) 
    mri.to_filename('Results/target.nii.gz')

    #Save input 0
    mri = nib.nifti1.Nifti1Image(test_dataset["images"][0].squeeze(0).cpu().detach().numpy(),None)  
    mri.to_filename('Results/input0.nii.gz')
    
    #Save input 1 
    mri = nib.nifti1.Nifti1Image(test_dataset["images"][1].squeeze(0).cpu().detach().numpy(),None)  
    mri.to_filename('Results/input1.nii.gz')

    #Save GT
    mri = nib.nifti1.Nifti1Image(((test_dataset["immGT"].squeeze(0).cpu().detach().numpy() - 2000) / 1000),None)  
    mri.to_filename('Results/GT.nii.gz')
    print(f"GT max: {mri.get_fdata().max()} \nGT min: {mri.get_fdata().min()}\n")
