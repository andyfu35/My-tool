import pandas as pd
import requests
from torch.nn.functional import interpolate
import xlwings as xw
import csv
from tqdm import tqdm
import time
import argparse
import os


for type in range(1 , 32):

    months_days = []
    i = 1

    if type <= 9:
        type = "0" + str(type)

    path = "股票/stock_data/" + str(type)

    if not os.path.isdir(path):
        os.mkdir(path)

    for month in range(1, 13):
        if month <= 9:
            month = "0" + str(month)
        for day in range(1, 32):
            if day <= 9:
                day  = "0" + str(day)
            month_day = str(month) + str(day)
            months_days.append(month_day)
    for date in tqdm(months_days):
        
        csv_path = "股票/stock_data/" + str(type) + "/2021" +str(date) + ".csv"
        url = "https://www.twse.com.tw/exchangeReport/MI_INDEX?response=csv&date=2021"+ str(date) + "&type=" + str(type)


        res = requests.get(url)
        data = res.text
        if len(data) == 0:
            i = i + 1
            pass
        else:
            cleaned_data = []
            for data_ in data.split('\n'):
                if len(data_.split('","')) == 16 and data_.split('","')[0][0] != '=':
                    cleaned_data.append([ele.replace('",\r', '').replace('"', '')
                                        for ele in data_.split('","')])

            with open(csv_path, 'w', newline="", encoding='utf-8') as file:
                mywriter = csv.writer(file, delimiter =" ")
                mywriter.writerows(cleaned_data)
                i = i + 1
        if i % 15 == 0:
            time.sleep(60)
    time.sleep(60)   






