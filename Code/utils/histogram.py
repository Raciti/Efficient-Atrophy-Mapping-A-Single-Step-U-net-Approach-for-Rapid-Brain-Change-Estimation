import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

img = nib.load('../../Results/GT.nii.gz')

data = img.get_fdata()

GT =  data[(data < -0.01) | (data > 0.01)]

# hist, bins = np.histogram(data_nonzero, bins=50)

plt.figure(figsize=(10, 6))
plt.hist(GT, bins=50, color='blue', edgecolor='black')
plt.title('Istogramma della GT')
plt.xlabel('Intensità')
plt.ylabel('Frequenza')
plt.grid(True)
plt.savefig("../../Results/histogram_GT.png")
plt.close()

img = nib.load('../../Results/target.nii.gz')

data = img.get_fdata()

Tg = data[(data < -0.01) | (data > 0.01)]


# hist, bins = np.histogram(data_nonzero, bins=50)

plt.figure(figsize=(10, 6))
plt.hist(Tg, bins=50, color='blue', edgecolor='black')
plt.title('Istogramma del Target')
plt.xlabel('Intensità')
plt.ylabel('Frequenza')
plt.grid(True)
plt.savefig("../../Results/histogram_Target.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.hist(GT, bins=50, color='blue', edgecolor='black', alpha=0.5, label='Immagine 1')
plt.hist(Tg, bins=50, color='red', edgecolor='black', alpha=0.5, label='Immagine 2')
plt.title('Istogrammi sovrapposti')
plt.xlabel('Intensità')
plt.ylabel('Frequenza')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig("../../Results/histograms_overlap.png")
plt.close()