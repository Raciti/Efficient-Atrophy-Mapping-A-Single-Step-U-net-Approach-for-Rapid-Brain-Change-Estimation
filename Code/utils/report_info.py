from pathlib import Path
import csv
from tqdm import tqdm



headers = ['MRIs ID', 'Area [mm^2]', 'Volc [mm^3]', 'Ratio [mm]', 'PBVC [%]']

csv_file = '../../Results/info_report.csv'

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)



directory = Path("xxxx") # Put the SIENA file directory

val = []

for patient in directory.iterdir():
    id_mris = str(patient).split('/')[-1]

    for file in patient.iterdir():
        name_file = str(file).split('/')[-1]
        
        if name_file == "report.siena":
            with open(file, 'r') as file:
                report = file.readlines()
            

            
            area = report[-6].split(" ")[-2]
            volc = report[-5].split(" ")[-2]
            ratio = report[-4].split(" ")[-2]
            PBVC = report[-3].split(" ")[-2]


            val.append([id_mris, area, volc, ratio, PBVC])




with open(csv_file, 'a', newline='') as file:
    writer = csv.writer(file)
    for row in val:
        writer.writerow(row)