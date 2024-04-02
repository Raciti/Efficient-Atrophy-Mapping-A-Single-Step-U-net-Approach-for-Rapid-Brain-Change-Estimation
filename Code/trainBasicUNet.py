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
from monai.transforms import LoadImage, Compose

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trasformazione personalizzata
class preprocessing:
    def __call__(self, data):
        image1 = LoadImage(dtype=torch.float32)(data['immA']).unsqueeze(0)
        image2 = LoadImage(dtype=torch.float32)(data['immB']).unsqueeze(0)
        ground_truth = LoadImage(dtype=torch.float32)(data['immGT']).unsqueeze(0)
        concatenated_image = torch.cat([image1, image2], dim=0)  # Concatena le immagini lungo l'asse dei canali
        return concatenated_image, ground_truth # Ritorna immagine concatenata e GT


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True, help='Path csv of train dataset.')
    parser.add_argument('--valid_csv', type=str, required=True, help='Path csv of valid dataset.')
    parser.add_argument('--dict_save_model', type=str, required=True, help='Path directory where to save checkpoint.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--valid_batch_size', type=int, default=8)

    args = parser.parse_args()



    transform = Compose([preprocessing()])

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

    optimizer = AdamW(UNet.parameters(), lr=1e-4)
    loss_function = nn.MSELoss(reduction='mean')

losses = []
for epoch in range(args.epochs):
    for mode in ['train', 'valid']:
        epoch_losses = []
        loader = loaders[mode]
        UNet.train() if mode == 'train' else UNet.eval()

        for batch in tqdm(loader):
            
    
            with torch.set_grad_enabled(mode == 'train'):
                with autocast(enabled=True):

                    data, GT = batch 
                    data = data.to(device)
                    GT = GT.to(device)
                    output = UNet(data)

                    if mode == 'train':
                        optimizer.zero_grad() 
                        loss = loss_function(output, GT) 
                        print(f"Epoch[{epoch}/{args.epochs}] --> mean MSE: {loss.item()}")
                        loss.backward()  
                        #epoch_losses.append(loss.item())
                        optimizer.step()

                    else:
                        loss = loss_function(output, GT)
                        epoch_losses.append(loss.item())

    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(avg_epoch_loss)
    if min(losses) == avg_epoch_loss:
        torch.save(UNet.state_dict(), os.path.join(args.dict_save_model, f'Unet-{epoch}.pth'))
        torch.save(optimizer.state_dict(), os.path.join(args.dict_save_model, f'optim-{epoch}.pth'))
