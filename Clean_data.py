import pandas as pd 
import os
import csv
import numpy as np
from tqdm import tqdm, trange

PATH = "股票/stock_data/N_totol/"
csv_name_ = "股票/totol.csv"
date_name = os.listdir(PATH)

for i in trange(len(date_name)):
    N_date = date_name[i]
    csv_name = str(PATH) + str(N_date)
    dataframe = pd.read_csv(csv_name, header=None, sep='\t')
    data_1 = []
    for iii in range(len(dataframe)-1):
        iii = iii + 1
        data_1.append(dataframe.values[iii, 0]) 
    if i <= 1:
        set_2 = set(data_1)
    else:
        set_1 = set(data_1)
        set_2 = set_1 & set_2
    
data_ = []
for i in range(len(date_name)):
    data = []
    N_date = date_name[i]
    csv_name = str(PATH) + str(N_date)
    dataframe = pd.read_csv(csv_name, header=None, sep='\t', encoding='utf8')
    dataframe.set_index(0,inplace = True)
    data.append(dataframe.loc["2603", 5])
    for ii in set_2:
        if ii == "2603":
            pass
        else:
            data.append(dataframe.loc[str(ii), 5])
    data_.append(data)

with open(csv_name_, 'w', newline="") as file:
    mywriter = csv.writer(file, delimiter ="\t")
    mywriter.writerows(data_)

