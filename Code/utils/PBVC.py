import os, sys, re, subprocess
import time
import csv
import nibabel as nib
import argparse
import numpy as np
from tqdm import tqdm
from statistics import mean
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
from monai.transforms import Resize, Compose

headers = ['MRIs ID', 'Corr','Area [mm^2]', 'Volc [mm^3]', 'Ratio [mm]', 'PBVC [%]', 'Target Volume [mm^3]','Target PBVC [%]', 'Diff Volume', 'Diff PBVC', 'SSIM']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, required=True,
    #                     help="Directory of file directorys") 
    parser.add_argument('--report', type=str, default="../../Results/info_report_testset.csv",
                        help="File whit values") 

    #cwd = os.getcwd()
    args = parser.parse_args()
    
    with open(args.report, 'r') as file:
                report = file.readlines()

    data = []
    for i in tqdm(range(1, len(report))):
        
        file_name = report[i].split(',')[0]
        corr = float(report[i].split(',')[1])
        Gt_PBVC = float(report[i].split(',')[-1])

        print(file_name, corr, Gt_PBVC)
        print(f"/storage/data_4T/lpuglisi-siena/SIENA/{file_name}/B_to_A_flow.nii.gz")

        start_time = time.time()
    
        calib = corr #corr da sostituire con quello corretto
        pbvc_list = []

        print("COMPUTE PBVC FOR B_to_A_flow")
        img_1 = nib.load(f"/storage/data_4T/lpuglisi-siena/SIENA/{file_name}/B_to_A_flow.nii.gz")
        img_1_edgepoint = nib.load(f"/storage/data_4T/lpuglisi-siena/SIENA/{file_name}/B_to_A_edgepoints.nii.gz").get_fdata()
        flow_1 = img_1.get_fdata()
        print(flow_1.shape)

        total = flow_1.sum()
        count= (img_1_edgepoint != 0).sum() #(flow_1 != 0).sum()

        # Get voxel dimensions
        voxel_dims = (img_1.header["pixdim"])[1:4]
        voxel_volume = abs( voxel_dims[0] * voxel_dims[1] *voxel_dims[2] )
        voxel_area = pow(voxel_volume,(0.6666667))

        area = count * voxel_area
        print(f"AREA:  {area} mm^2")
        volume = total * voxel_volume
        print(f"VOLUME:  {round(volume,1)} mm^3")
        ratio = (total*voxel_volume) / (count*voxel_area)
        print(f"RATIO:  {round(ratio,6)} mm")

        ratio = (total * voxel_volume) / (count * voxel_area)
        PBVC =  (calib * 30 * total * voxel_volume) / (count * voxel_area)
        print(f"PBVC = {round(PBVC,4)} %")
        pbvc_list.append(PBVC)
    
#------------------------------------------------------------------------
        print("")
        print("COMPUTE PBVC FOR A_to_B_flow")
        img_GT = nib.load(f"/storage/data_4T/lpuglisi-siena/SIENA/{file_name}/A_to_B_flow.nii.gz").get_fdata()
        img_pred = nib.load(f"/storage/data_4T/riccardoraciti/dataset/{file_name}_flow.nii.gz")
        img_pred_edgepoint = nib.load(f"/storage/data_4T/lpuglisi-siena/SIENA/{file_name}/A_to_B_edgepoints.nii.gz").get_fdata()
        flow_pred = img_pred.get_fdata()

        #RESIZE to original dim
        flow_resize = np.expand_dims(flow_pred, 0)
        transform = Compose([Resize(spatial_size= img_GT.shape )]) #modificato 
        flow_resize = transform(flow_resize)
        flow_resize = flow_resize.squeeze()
        flow_resize = np.array(flow_resize)
        print(img_GT.shape)
        print(flow_resize.shape)
        print(img_pred_edgepoint.shape)

        SSIM = ssim(img_GT, flow_resize, data_range= flow_resize.max() - flow_resize.min())
        print("SSIM: ", SSIM)

        total = flow_resize.sum()
        count= (img_pred_edgepoint != 0).sum() #(flow_pred != 0).sum()
        
        # Get voxel dimensions
        voxel_dims = (img_pred.header["pixdim"])[1:4]
        voxel_volume = abs( voxel_dims[0] * voxel_dims[1] *voxel_dims[2] )

        print(voxel_volume)
        voxel_area = pow(voxel_volume,(0.6666667))
        print(voxel_area)

        area = count * voxel_area
        print(f"AREA:  {area} mm^2")
        volume = total * voxel_volume
        print(f"VOLUME:  {round(volume,1)} mm^3")
        ratio = (total*voxel_volume) / (count*voxel_area)
        print(f"RATIO:  {round(ratio,6)} mm")

        ratio = (total * voxel_volume) / (count * voxel_area)
        PBVC =  (calib * 30 * total * voxel_volume) / (count * voxel_area)
        print(f"PBVC = {round(PBVC,4)} %")

        pbvc_list.append(abs(PBVC))

        print("")
        print(f"finalPBVC: -{mean(pbvc_list)} %")

        print("--- %s seconds ---" % (time.time() - start_time))
        data.append([report[i].split(',')[0], float(report[i].split(',')[1]), int(report[i].split(',')[2]), float(report[i].split(',')[3]), float(report[i].split(',')[4]),float(report[i].split(',')[5]),
                    round(volume,1), float(f"-{round(mean(pbvc_list),4)}"), round(float(report[i].split(',')[3]) - volume, 2), round(float(report[i].split(',')[5]) - float(f"-{mean(pbvc_list)}"), 4), round(SSIM, 3)])
        

csv_file = '../../Results/info_report_testSet_whit_target.csv'

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for row in data:
        writer.writerow(row)