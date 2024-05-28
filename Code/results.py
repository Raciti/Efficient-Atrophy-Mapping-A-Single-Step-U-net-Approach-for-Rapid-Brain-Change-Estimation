import os
import argparse

import torch
from torch.utils.data import DataLoader

from monai.data import CSVDataset
from monai.networks.nets import BasicUNet
from monai.transforms import Compose, LoadImageD, ToTensord,  EnsureChannelFirstD,  ResizeD 
from torchmetrics.image import StructuralSimilarityIndexMeasure

from tqdm import tqdm
from utils import losses

from utils import transform

import matplotlib.pyplot as plt
from monai.losses import LocalNormalizedCrossCorrelationLoss as NN

import time

import pickle


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
    transform.preprocessing(['immA', 'immB']),
])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--calculate', type=bool, default=False,
                        help='Calculate the dicts.')
    
    args = parser.parse_args()

    if args.calculate == True:
        test_dataset = CSVDataset(src="Data/test.csv", transform=transform, skiprows=1) 
        
        train_loader = DataLoader(dataset=test_dataset,
                                batch_size=1,
                                shuffle=False)
        

        Time = {'cuda': [], 'cpu': []}
        MSE = {'cuda': [], 'cpu': []}
        SSIM = {'cuda': [], 'cpu': []}
        NCC = {'cuda': [], 'cpu': []}
        mse = losses.CustomMSELoss(2, "max")
        ncc = NN(reduction="mean")

        for device in ['cuda', 'cpu']:
            print(device)
            UNet = BasicUNet(spatial_dims=3, in_channels= 2, out_channels = 1, features=(32, 32, 64, 128, 256, 32)).to(device)
            UNet.load_state_dict(torch.load("Model/Unet.pth"))
            ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

            for batch in tqdm(train_loader):
                data, gt = batch["images"], batch["immGT"]
                
                data = data.to(device)
                gt = gt.to(device)
                
                start_time = time.time()
                output = UNet(data)
                end_time = time.time()

                Time[device].append(round(end_time-start_time,2))

                target = (output - 2000) / 1000
                gt = (gt - 2000) / 1000

                MSE[device].append(mse(target, gt).item())

                SSIM[device].append(ssim(target,gt).item())

                NCC[device].append(ncc(target,gt).item())
            

        with open('Results/Time.pkl', 'wb') as file:
            pickle.dump(Time, file)
        with open('Results/MSE.pkl', 'wb') as file:
            pickle.dump(MSE, file)
        with open('Results/SSIM.pkl', 'wb') as file:
            pickle.dump(SSIM, file)
        with open('Results/NCC.pkl', 'wb') as file:
            pickle.dump(NCC, file)

    else:
        with open("Results/Time.pkl", 'rb') as file:
            Time = pickle.load(file)
        with open("Results/MSE.pkl", 'rb') as file:
            MSE = pickle.load(file)
        with open("Results/SSIM.pkl", 'rb') as file:
            SSIM = pickle.load(file)
        with open("Results/NCC.pkl", 'rb') as file:
            NCC = pickle.load(file)

    
    plt.title("Esecution Time")
    plt.xlabel('Iterations')
    plt.ylabel('Time')
    plt.plot(Time["cuda"], label='gpu')
    plt.plot(Time["cpu"], label='cpu')
    plt.legend()
    plt.savefig("Results/Time.png")
    plt.close()
    
    plt.title("Esecution Time GPU")
    plt.xlabel('Iterations')
    plt.ylabel('Time')
    plt.plot(Time["cuda"][1:], label='gpu')
    plt.legend()
    plt.savefig("Results/Time_gpu.png")
    plt.close()
    
    max_time_gpu = max(Time["cuda"])
    min_time_gpu = min(Time["cuda"])
    mean_time_gpu = sum(Time["cuda"]) / len(Time["cuda"])
    print(f"Time GPU -> max:{max_time_gpu}; min:{min_time_gpu}; mean:{mean_time_gpu}.")
    
    plt.title("Esecution Time CPU")
    plt.xlabel('Iterations')
    plt.ylabel('Time')
    plt.plot(Time["cpu"], label='cpu')
    plt.legend()
    plt.savefig("Results/Time_cpu.png")
    plt.close()

    max_time_cpu = max(Time["cpu"])
    min_time_cpu = min(Time["cpu"])
    mean_time_cpu = sum(Time["cpu"]) / len(Time["cpu"])
    
    print(f"Time CPU -> max:{max_time_cpu}; min:{min_time_cpu}; mean:{mean_time_cpu}.")

    plt.title("MSE")
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.plot(MSE["cpu"], label='cpu')
    plt.plot(MSE["cuda"], label='gpu')
    plt.legend()
    plt.savefig("Results/MSE.png")
    plt.close()

    max_mse_cpu = max(MSE["cpu"])
    min_mse_cpu = min(MSE["cpu"])
    mean_mse_cpu = sum(MSE["cpu"]) / len(MSE['cpu'])

    max_mse_gpu = max(MSE["cuda"])
    min_mse_gpu = min(MSE["cuda"])
    mean_mse_gpu = sum(MSE["cuda"]) / len(MSE['cuda'])

    print(f"MSE CPU -> max:{max_mse_cpu}; min:{min_mse_cpu}; mean:{mean_mse_cpu}.")
    print(f"MSE GPU -> max:{max_mse_gpu}; min:{min_mse_gpu}; mean:{mean_mse_gpu}.")

    plt.title("SSIM")
    plt.xlabel('Iterations')
    plt.ylabel('SSIM Loss')
    plt.plot(SSIM["cpu"], label='cpu')
    plt.plot(SSIM["cuda"], label='gpu')
    plt.legend()
    plt.savefig("Results/SSIM.png")
    plt.close()

    max_ssim_cpu = max(SSIM["cpu"])
    min_ssim_cpu = min(SSIM["cpu"])
    mean_ssim_cpu = sum(SSIM["cpu"]) / len(SSIM['cpu'])

    max_ssim_gpu = max(SSIM["cuda"])
    min_ssim_gpu = min(SSIM["cuda"])
    mean_ssim_gpu = sum(SSIM["cuda"]) / len(SSIM['cuda'])

    print(f"SSIM CPU -> max:{max_ssim_cpu}; min:{min_ssim_cpu}; mean:{mean_ssim_cpu}.")
    print(f"SSIM GPU -> max:{max_ssim_gpu}; min:{min_ssim_gpu}; mean:{mean_ssim_gpu}.")


    plt.title("LNCC")
    plt.xlabel('Iterations')
    plt.ylabel('LNCC Loss')
    plt.plot(NCC["cpu"], label='cpu')
    plt.plot(NCC["cuda"], label='gpu')
    plt.legend()
    plt.savefig("Results/NCC.png")
    plt.close()

    max_ncc_cpu = max(NCC["cpu"])
    min_ncc_cpu = min(NCC["cpu"])
    mean_ncc_cpu = sum(NCC["cpu"]) / len(NCC['cpu'])

    max_ncc_gpu = max(NCC["cuda"])
    min_ncc_gpu = min(NCC["cuda"])
    mean_ncc_gpu = sum(NCC["cuda"]) / len(NCC['cuda'])

    print(f"NCC CPU -> max:{max_ncc_cpu}; min:{min_ncc_cpu}; mean:{mean_ncc_cpu}.")
    print(f"NCC GPU -> max:{max_ncc_gpu}; min:{min_ncc_gpu}; mean:{mean_ncc_gpu}.")

# Time GPU -> max:2.94; min:0.02; mean:0.04949494949494931.
# Time CPU -> max:3.19; min:2.67; mean:2.9836363636363634.

# MSE CPU -> max:0.0115367341786623; min:9.864193998510018e-05; mean:0.0008173479576303501.
# MSE GPU -> max:0.011536423116922379; min:9.863869490800425e-05; mean:0.0008173438736132929.

# SSIM CPU -> max:0.9432607889175415; min:0.7165855169296265; mean:0.884942119771784.
# SSIM GPU -> max:0.943253755569458; min:0.7165293097496033; mean:0.8849537438816495.

# NCC CPU -> max:-0.10716778039932251; min:-0.17413167655467987; mean:-0.1486158450745573.
# NCC GPU -> max:-0.10717373341321945; min:-0.17412962019443512; mean:-0.14861157137637188.