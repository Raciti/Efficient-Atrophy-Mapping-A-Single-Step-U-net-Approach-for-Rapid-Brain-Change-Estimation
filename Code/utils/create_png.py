import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import argparse
import torch
    

   
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--MRI', type=bool, default = False)
    parser.add_argument('--MRI_pre', type=bool, default = False)
    parser.add_argument('--MRI_Skull_Stripped', type=bool, default = False)
    parser.add_argument('--A_B', type=bool, default = False)
    parser.add_argument('--GT', type=bool, default = False)
    parser.add_argument('--GT_Target', type=bool, default = False)
    parser.add_argument('--GT_Target_full', type=bool, default = False)
    

    args = parser.parse_args()
   
    if args.MRI == True:
        nii_img = nib.load('Data/A.nii.gz')
        img_data = nii_img.get_fdata()

        x_center = img_data.shape[0] // 2
        y_center = img_data.shape[1] // 2
        z_center = img_data.shape[2] // 2

        combined_image = plt.figure()

        plt.subplot(1, 3, 3)
        plt.imshow(np.rot90(img_data[x_center, :, :]), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(np.rot90(img_data[:, y_center, :]), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 1)
        plt.imshow(np.rot90(img_data[:, :, z_center]), cmap='gray')
        plt.axis('off')

        plt.savefig('Results/MRI.png', bbox_inches='tight', pad_inches=0, facecolor='black')

        plt.close()

    if args.MRI_pre == True:
        nii_img = nib.load('Data/A.nii.gz')
        A = nii_img.get_fdata()

        nii_img = nib.load('Data/B.nii.gz')
        B = nii_img.get_fdata()

        slice_index = A.shape[2] // 2

        axial_slice = np.rot90(A[:, :, slice_index])

        plt.imshow(axial_slice, cmap='gray')
        plt.axis('off')

        plt.savefig('Results/A_pre.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        slice_index = B.shape[2] // 2

        axial_slice = np.rot90(B[:, :, slice_index])

        plt.imshow(axial_slice, cmap='gray')
        plt.axis('off')

        plt.savefig('Results/B_pre.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    
    if args.MRI_Skull_Stripped == True:
        nii_img = nib.load('Data/A_halfwayto_B_brain.nii.gz')
        img_data = nii_img.get_fdata()

        x_center = img_data.shape[0] // 2
        y_center = img_data.shape[1] // 2
        z_center = img_data.shape[2] // 2

        combined_image = plt.figure()

        plt.subplot(1, 3, 3)
        plt.imshow(np.rot90(img_data[x_center, :, :]), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(np.rot90(img_data[:, y_center, :]), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 1)
        plt.imshow(np.rot90(img_data[:, :, z_center]), cmap='gray')
        plt.axis('off')

        plt.savefig('Results/MRI_skullStripped.png', bbox_inches='tight', pad_inches=0, facecolor='black')

        plt.close()

    if args.A_B == True:
        nii_img = nib.load('Results/input0.nii.gz')
        A = nii_img.get_fdata()

        nii_img = nib.load('Results/input1.nii.gz')
        B = nii_img.get_fdata()

        slice_index = A.shape[2] // 2

        axial_slice = np.rot90(A[:, :, slice_index])

        plt.imshow(axial_slice, cmap='gray')
        plt.axis('off')

        plt.savefig('Results/A.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        slice_index = B.shape[2] // 2

        axial_slice = np.rot90(B[:, :, slice_index])

        plt.imshow(axial_slice, cmap='gray')
        plt.axis('off')

        plt.savefig('Results/B.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    if args.GT_Target == True:
        nii_img = nib.load('Results/GT.nii.gz')
        GT = nii_img.get_fdata()

        nii_img = nib.load('Results/target.nii.gz')
        target = nii_img.get_fdata()

        slice_index = GT.shape[2] // 2

        axial_slice = np.rot90(GT[:, :, slice_index])

        plt.imshow(axial_slice, cmap='gray')
        plt.axis('off')

        plt.savefig('Results/GT.png', bbox_inches='tight', pad_inches=0, facecolor='white')
        plt.close()

        slice_index = target.shape[2] // 2

        axial_slice = np.rot90(target[:, :, slice_index])

        plt.imshow(axial_slice, cmap='gray')
        plt.axis('off')

        plt.savefig('Results/target.png', bbox_inches='tight', pad_inches=0, facecolor='white')
        plt.close()
    
    if args.GT == True:
        nii_img = nib.load('Data/GT.nii.gz')
        img_data = nii_img.get_fdata()

        x_center = img_data.shape[0] // 2
        y_center = img_data.shape[1] // 2
        z_center = img_data.shape[2] // 2

        combined_image = plt.figure()

        

        plt.subplot(1, 3, 1)
        plt.imshow(np.rot90(img_data[:, :, z_center]), cmap='gray')
        plt.axis('off')

        plt.savefig('Results/GT.png', bbox_inches='tight', pad_inches=0, facecolor='grey')

        plt.close()

    if args.GT_Target_full == True:
        nii_img = nib.load('Results/GT.nii.gz')
        img_data = nii_img.get_fdata()

        x_center = img_data.shape[0] // 2
        y_center = img_data.shape[1] // 2
        z_center = img_data.shape[2] // 2


        plt.imshow(np.rot90(img_data[x_center, :, :]), cmap='gray')
        plt.axis('off')
        plt.savefig('Results/GT_S.png', bbox_inches='tight', pad_inches=0)
        plt.close()


        plt.imshow(np.rot90(img_data[:, y_center, :]), cmap='gray')
        plt.axis('off')
        plt.savefig('Results/GT_C.png', bbox_inches='tight', pad_inches=0)
        plt.close()


        plt.imshow(np.rot90(img_data[:, :, z_center]), cmap='gray')
        plt.axis('off')
        plt.savefig('Results/GT_A.png', bbox_inches='tight', pad_inches=0)
        plt.close()



        nii_img = nib.load('Results/target.nii.gz')
        img_data = nii_img.get_fdata()
        img_data = torch.from_numpy(img_data).squeeze(0).squeeze(0)
        img_data = img_data.numpy()


        x_center = img_data.shape[0] // 2
        y_center = img_data.shape[1] // 2
        z_center = img_data.shape[2] // 2

        plt.imshow(np.rot90(img_data[x_center, :, :]), cmap='gray')
        plt.axis('off')
        plt.savefig('Results/Target_S.png', bbox_inches='tight', pad_inches=0)
        plt.close()


        plt.imshow(np.rot90(img_data[:, y_center, :]), cmap='gray')
        plt.axis('off')
        plt.savefig('Results/Target_C.png', bbox_inches='tight', pad_inches=0)
        plt.close()


        plt.imshow(np.rot90(img_data[:, :, z_center]), cmap='gray')
        plt.axis('off')
        plt.savefig('Results/Target_A.png', bbox_inches='tight', pad_inches=0)
        plt.close()