import os
import argparse

import torch
from torch.utils.data import DataLoader

import monai
from monai.data import CSVDataset
from monai.networks.nets import BasicUNet
from monai.transforms import Compose, LoadImageD, ToTensord, EnsureChannelFirstD, ResizeD  
from tqdm import tqdm
from utils import transform

import nibabel as nib
    


basic_1 = Compose([
    LoadImageD(keys=["immA", "immB"]),
    EnsureChannelFirstD(keys=["immA", "immB"], channel_dim = 'no_channel'),
    ResizeD(keys=["immA", "immB"], spatial_size=(121, 145, 113)),
    ToTensord(keys=["immA", "immB"]),

])

transform = Compose([
    basic_1,
    transform.preprocessing_use(['immA', 'immB']),
])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_csv', type=str, required=True, 
                        help='Path csv of dataset.')
    parser.add_argument('--dict_save_flow', type=str, required=True, 
                        help='Path directory where to save flows.')
    
    
    args = parser.parse_args()

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    dataset_csv = CSVDataset(src=args.dataset_csv, transform=transform, skiprows=1) 
    

    loader = DataLoader(dataset=dataset_csv,
                            batch_size= 1,
                            shuffle=False)

  
    UNet = BasicUNet(spatial_dims=3, in_channels= 2, out_channels = 1, features=(32, 32, 64, 128, 256, 32)).to(device)
    UNet.load_state_dict(torch.load("../Model/Unet.pth"))

    i = 0
    for batch in tqdm(loader):

        data = batch["images"]
        data = data.to(device)
        
        output = UNet(data)

        target = (output.cpu().detach().numpy()  - 2000) / 1000

        #Save output
        flow = nib.nifti1.Nifti1Image(target, None) 
        flow.to_filename(os.path.join(args.dict_save_flow, dataset_csv.data[i]['immA'].split('/')[-2] + "_flow.nii.gz"))
        i += 1
                
        

