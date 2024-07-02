import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

file_path = '../../Results/info_report_testSet_whit_target.csv'  # Change this path to the actual file location
df = pd.read_csv(file_path)


mean_values = df[['Diff Volume', 'Diff PBVC', 'SSIM']].mean()
std_values = df[['Diff Volume', 'Diff PBVC', 'SSIM']].std()

print("Mean values:")
print(mean_values)
print("\nStandard deviation values:")
print(std_values)

# Mean values:
# Diff Volume    3238.427879
# Diff PBVC        -0.217898
# SSIM              0.796263

# Standard deviation values:
# Diff Volume    7221.052499
# Diff PBVC         1.494223
# SSIM              0.023821


plt.figure(figsize=(10, 6))
plt.plot(df['PBVC [%]'], label='PBVC [%]', marker='o')
plt.plot(df['Target PBVC [%]'], label='Target PBVC [%]', marker='x')

plt.title('Valori PBVC e Target PBVC')
plt.xlabel('Indice')
plt.ylabel('Valore [%]')

plt.legend()

plt.grid(True)
plt.savefig("../../Results/PBVC.png")
plt.close()

corr, _ = pearsonr(df['PBVC [%]'], df['Target PBVC [%]'])

plt.figure(figsize=(10, 6))
plt.scatter(df['PBVC [%]'], df['Target PBVC [%]'], alpha=0.6)

m, b = np.polyfit(df['PBVC [%]'], df['Target PBVC [%]'], 1)
plt.plot(df['PBVC [%]'], m * df['PBVC [%]'] + b, color='red')

plt.title(f'Scatter Plot e Correlazione di Pearson (r = {corr:.2f})')
plt.xlabel('PBVC [%]')
plt.ylabel('Target PBVC [%]')

plt.grid(True)
plt.savefig("../../Results/PBVC_pearson.png")
plt.close()