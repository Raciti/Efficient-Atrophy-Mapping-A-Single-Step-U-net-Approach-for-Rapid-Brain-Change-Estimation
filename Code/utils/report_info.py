from pathlib import Path
import csv
from tqdm import tqdm



headers = ['MRIs ID', 'Corr','Area [mm^2]', 'Volc [mm^3]', 'Ratio [mm]', 'PBVC [%]']

csv_file = '../../Results/info_report.csv'

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)



directory = Path("XXXX") # Put the SIENA file directory

val = []

for patient in directory.iterdir():
    id_mris = str(patient).split('/')[-1]

    for file in patient.iterdir():
        name_file = str(file).split('/')[-1]
        
        if name_file == "report.siena":
            with open(file, 'r') as file:
                report = file.readlines()
            

            crr = float((report[15].split("corr=")[1])[0:-1])
            area = report[-6].split(" ")[-2]
            volc = report[-5].split(" ")[-2]
            ratio = report[-4].split(" ")[-2]
            PBVC = report[-3].split(" ")[-2]


            val.append([id_mris, crr, area, volc, ratio, PBVC])




with open(csv_file, 'a', newline='') as file:
    writer = csv.writer(file)
    for row in val:
        writer.writerow(row)    

with open("../../Results/info_report.csv", 'r') as file:
                report = file.readlines()

with open("../../Data/test.csv", 'r') as file:
                test_set = file.readlines()



data = []
for i in range(1, len(test_set)):
    id = test_set[i].split(',')[0].split('/')[5]
        
    for j in range(1 , len(report)):

        if report[j].split(',')[0] == id:
            data.append([report[j].split(',')[0], float(report[j].split(',')[1]), int(report[j].split(',')[2]), float(report[j].split(',')[3]), float(report[j].split(',')[4]),float( report[j].split(',')[5])])

    
csv_file = '../../Results/info_report_testset.csv'

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for row in data:
        writer.writerow(row)