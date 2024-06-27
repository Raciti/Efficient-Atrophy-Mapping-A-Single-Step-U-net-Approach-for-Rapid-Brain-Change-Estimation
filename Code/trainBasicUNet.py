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
from monai.transforms import Compose, LoadImageD, ToTensord, EnsureChannelFirstD, ResizeD  
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
import numpy as np
from utils import losses
from monai.losses import LocalNormalizedCrossCorrelationLoss as NN
from utils import transform


import matplotlib.pyplot as plt


basic_1 = Compose([
    LoadImageD(keys=["immA", "immB", "immGT"]),
    EnsureChannelFirstD(keys=["immA", "immB", "immGT"], channel_dim = 'no_channel'),
    ResizeD(keys=["immA", "immB", "immGT"], spatial_size=(182, 218, 170)),
    ToTensord(keys=["immA", "immB", "immGT"]),

])

transform = Compose([
    basic_1,
    transform.preprocessing(['immA', 'immB']),
])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True, 
                        help='Path csv of train dataset.')
    parser.add_argument('--valid_csv', type=str, required=True, 
                        help='Path csv of valid dataset.')
    parser.add_argument('--dict_save_model', type=str, required=True, 
                        help='Path directory where to save checkpoint.')
    parser.add_argument('--gpu', type=str, default="0", 
                        help='GPU ID numbers (default: 0).')
    parser.add_argument('--loss', type=str, default="mse", 
                        help='Image reconstruction loss - can be mse or ncc (default: mse).')
    parser.add_argument('--reduction', type=str, default="mean", 
                        help='Reduction loss - can be sum, max or mean. max can only be used in mse loss. (default: mean).')
    parser.add_argument('--exp', type=int, default=2, 
                        help='Exponent for the calculation of mse (default: mean).')
    parser.add_argument('--load_model', type = str, default= "",
                        help='Load a version of the previous model to continue training it. (need unet.pth and optim.pth)')
    parser.add_argument('--load_optim', type = str, default= "",
                        help='Load a version of the previous optim to continue training it. (need unet.pth and optim.pth)')
    parser.add_argument('--scheduler', type = bool, default= False,
                        help='Enable scheduler usage')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)' )
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='Batch size training (default: 4)')
    parser.add_argument('--valid_batch_size', type=int, default=4,
                        help='Batch size validation (default: 4)')
    
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    train_dataset = CSVDataset(src=args.train_csv, transform=transform, skiprows=1) 
    valid_dataset = CSVDataset(src=args.valid_csv, transform=transform, skiprows=1)

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=args.train_batch_size,
                            shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=args.valid_batch_size, 
                            shuffle=True)

    loaders = { 'train': train_loader, 'valid': valid_loader }


    UNet = BasicUNet(spatial_dims=3, in_channels= 2, out_channels = 1, features=(32, 32, 64, 128, 256, 32)).to(device)
    optimizer = AdamW(UNet.parameters(), lr=0.001)
    
    if args.load_model != '' and args.load_optim != '':
        UNet.load_state_dict(torch.load(args.load_model))
        optimizer.load_state_dict(torch.load(args.load_optim))
    
    if (args.load_model != '') != (args.load_optim != ''):
        raise RuntimeError("Both the path of the model to be trained and the path of the relative Optim must be provided.")

    if args.scheduler == True:    
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                                T_max = len(train_loader) * args.epochs, 
                                                                eta_min = 1e-5)

    if args.loss == "mse" and args.reduction == "max":
        loss_function = losses.CustomMSELoss(args.exp, "max")
    elif args.loss == "mse" and args.reduction == "mean":
        loss_function = losses.CustomMSELoss(args.exp, "mean")
    elif args.loss == "mse" and args.reduction == "sum":
        loss_function = losses.CustomMSELoss(args.exp, "sum")
    elif args.loss == "ncc" and args.reduction == "mean":
        loss_function = NN()
    elif args.loss == "ncc" and args.reduction == "sum":
        loss_function = NN(reduction="sum")
    elif args.loss == "ncc" and args.reduction == "max":
        raise RuntimeError("The value of reduction is not valid. max can only be used in mse loss.")
        
    else:
        raise RuntimeError("The value of loss or reduction is not valid.")
    
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    history_loss = {"train": [], "valid": []}
    history_acc = {"train": [], "valid": []}
    
    best_val_loss = None
    best_val_accuracy=0
    
    for epoch in range(args.epochs):
        sum_loss = {"train": 0, "valid": 0}
        sum_accuracy = {"train": 0, "valid": 0}
        
        for mode in ['train', 'valid']:
            loader = loaders[mode]
            UNet.train() if mode == 'train' else UNet.eval()

            for batch in tqdm(loader):
                with torch.set_grad_enabled(mode == 'train'):
                    with autocast(enabled=True):

                        data, GT = batch["images"], batch["immGT"]
                        data = data.to(device)
                       
                        GT = GT.to(device)

                        optimizer.zero_grad()

                        output = UNet(data)

                        loss = loss_function(output, GT)  
                        
                        # Update loss
                        sum_loss[mode] += loss.item()

                        if mode == 'train' and args.scheduler == True:
                            loss.backward()  
                            optimizer.step()
                            scheduler.step()
                            print(f"Current learning rate: {scheduler.get_last_lr()}")
                        elif mode == 'train':
                            loss.backward()  
                            optimizer.step()
                        
                        #compute accuracy
                        batch_accuracy = ssim(output, GT).sum().item()
                        sum_accuracy[mode] += batch_accuracy



        #avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_loss = {split: sum_loss[split]/len(loaders[split]) for split in ["train", "valid"]}
        epoch_acc = {split: sum_accuracy[split]/len(loaders[split]) for split in ["train", "valid"]}
        print("-" * 100)
        print("Epoch {}/{}".format(epoch, args.epochs))
        print(f"TRAIN LOSS MSE: {epoch_loss['train']:.4f}")
        print(f"VAL LOSS MSE: {epoch_loss['valid']:.4f}")
        print(f"TRAIN SSIM: {epoch_acc['train']:.4f}")
        print(f"VAL SSIM: {epoch_acc['valid']:.4f}")
        
        # Update history
        for split in ["train", "valid"]:
            history_loss[split].append(epoch_loss[split])
            history_acc[split].append(epoch_acc[split])
        
        # Check if we obtained the best 
        if(best_val_loss is None or best_val_loss > epoch_loss["valid"]):
        #if min(losses) == avg_epoch_loss:
            best_val_loss=epoch_loss["valid"]
            print(f"Best val loss: {best_val_loss:.4f}\n")
            torch.save(UNet.state_dict(), os.path.join(args.dict_save_model, f'Unet-{epoch}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.dict_save_model, f'optim-{epoch}.pth'))
        
    # Plot loss history
    plt.title("MSE")
    plt.xlabel('Epoch')
    plt.ylabel('MSELoss')
    plt.plot(history_loss["train"], label='train')
    plt.plot(history_loss["valid"], label='val')
    plt.legend()
    plt.savefig(os.path.join(args.dict_save_model, 'MSELoss_numepoche_{}.png'.format(args.epochs)))
    plt.close()

    # Plot acc history
    plt.title("SSIM")
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.plot(history_acc["train"], label='train')
    plt.plot(history_acc["valid"], label='val')
    plt.legend()
    plt.savefig(os.path.join(args.dict_save_model, 'SSIM_numepoche_{}.png'.format(args.epochs)))
    plt.close()
