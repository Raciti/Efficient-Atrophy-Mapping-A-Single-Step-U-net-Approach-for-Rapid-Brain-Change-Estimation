import os, sys, re, subprocess
import time
import nibabel as nib
import argparse
import numpy as np
from statistics import mean
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
from monai.transforms import Resize, Compose

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='/storage/data_4T/alessia_medical_datasets/SIENA_output/I297035_to_I1174244') 

if __name__ == '__main__':
    
    cwd = os.getcwd()
    args = parser.parse_args()

    start_time = time.time()
    
    calib = 1.8672825298 #corr da sostituire con quello corretto
    pbvc_list = []

    print("COMPUTE PBVC FOR B_to_A_flow")
    img_1 = nib.load(os.path.join(args.data, 'B_to_A_flow.nii.gz'))
    img_1_edgepoint = nib.load(os.path.join(args.data, 'B_to_A_edgepoints.nii.gz')).get_fdata()
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
    img_GT = nib.load(os.path.join(args.data, 'A_to_B_flow.nii.gz')).get_fdata()
 
    img_pred = nib.load(os.path.join(args.data, 'target.nii.gz'))
    img_pred_edgepoint = nib.load(os.path.join(args.data, 'A_to_B_edgepoints.nii.gz')).get_fdata()
    flow_pred = img_pred.get_fdata()

    #RESIZE to original dim
    flow_resize = np.expand_dims(flow_pred, 0)
    transform = Compose([Resize(spatial_size=(182, 218, 170))])
    flow_resize = transform(flow_resize)
    flow_resize = flow_resize.squeeze()
    flow_resize = np.array(flow_resize)


    ssim = ssim(img_GT, flow_resize, data_range=flow_resize.max() - flow_resize.min())
    print("SSIM: ", ssim)

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