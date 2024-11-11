from F import PNL_per_day, DumpCSV_and_MesToTele, position_input, position_report
import pandas as pd
import datetime
import numpy as np
import time
import requests
from time import sleep
from ta.volume import MFIIndicator 
from ta.momentum import RSIIndicator, AwesomeOscillatorIndicator
from ta.trend import MACD, CCIIndicator
from ta.volatility import BollingerBands

#IMPORT HÀM GET_DATA
import sys
sys.path.append('F:\Data\API_Chien')
from get_data_vn import trading_day, get_data_ps_adjusted

# Low-pass filter
from scipy.signal import butter,lfilter 
def lowpass_filter(signal, ratio):
    b, a = butter(1, ratio, btype='low', analog=False)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

token = '6924463892:AAE05XWY1kuwOZ7u9XfHKEYMTEsf-mskcQY'
id = '-4075432474'
author = "DatNT"

#CÁC LIST TIME THƯỜNG DÙNG CHO CÁC KHUNG GIỜ SỬ DỤNG DATA 5P, 10P, 15P, 20P
list_time_5 = ['09:04:55', '09:09:55', '09:14:55', '09:19:55', 
             '09:24:55', '09:29:55', '09:34:55', '09:39:55', '09:44:55', '09:49:55', 
             '09:54:55', '09:59:55', '10:04:55', '10:09:55', '10:14:55', '10:19:55', 
             '10:24:55', '10:29:55', '10:34:55', '10:39:55', '10:44:55', '10:49:55', 
             '10:54:55', '10:59:55', '11:04:55', '11:09:55', '11:14:55', '11:19:55', 
             '11:24:55', '11:29:55', '13:04:55', '13:09:55', '13:14:55', '13:19:55', 
             '13:24:55', '13:29:55', '13:34:55', '13:39:55', '13:44:55', '13:49:55', 
             '13:54:55', '13:59:55', '14:04:55', '14:09:55', '14:14:55', '14:19:55', 
             '14:24:55', '14:27:55', '14:59:55']

list_time_10 = ['09:09:55', '09:19:55', '09:29:55', '09:39:55', '09:49:55', '09:59:55', 
             '10:09:55', '10:19:55', '10:29:55', '10:39:55', '10:49:55', '10:59:55', '11:09:55', 
             '11:19:55', '11:29:55', '13:09:55', '13:19:55', '13:29:55', 
             '13:39:55', '13:49:55', '13:59:55', '14:09:55', '14:19:55', '14:27:55', '14:59:55']

list_time_15 = ['09:14:55', '09:29:55', '09:44:55', '09:59:55', '10:14:55', '10:29:55', '10:44:55', '10:59:55', '11:14:55', '11:29:55', 
             '13:14:55', '13:29:55', '13:44:55', '13:59:55', '14:14:55', '14:27:55', '14:59:55']

list_time_20 = ['09:19:55', '09:39:55', '09:59:55', '10:19:55', '10:39:55', '10:59:55', 
             '11:19:55', '13:19:55', '13:39:55', '13:59:55', '14:19:55', '14:59:55']


#path to expiration_date file
df = pd.read_csv('/home/fin/dat_nt/Dragon/expiration_date.csv') 
df['Date'] = pd.to_datetime(df['Date'])
expiration_date = set(map(lambda x: x.date(), df['Date']))



#Alpha function:
def alpha(data, params):
    data['Close_filtered'] = pd.Series(lowpass_filter(data['Close'], params['t1']))#0.75
    data['MFI'] = MFIIndicator(high = data['High'], low = data['Low'], close = data['Close_filtered'], volume = data['Volume']).money_flow_index()
    #
    pos = 0
    position = []
    for i in range(len(data)):
        #
        MFI = data['MFI'].values[i]
        #
        if pos == 0:
            if MFI < params['t2']:#13
                pos = -1
            if MFI > params['t3']:#81
                pos = 1
        elif pos == -1:
            if MFI > params['t4']:#59
                pos = 0
        else:
            if MFI < params['t5']:#37
                pos = 0
        if data['Date'].iloc[i].time() == datetime.time(14, 25):
            if pos == - 1:
                pos = 0
        if data['Date'].iloc[i].time() == datetime.time(14, 45):
            if len(position) > 0:
                if data['Date'].iloc[i].date() in expiration_date:
                    pos = 0

        position.append(pos)
    data['pos'] = position
    return data



while True: 
    time_now = datetime.datetime.now()
    if time_now.weekday == 3 and 14 < time_now.day < 22:   #NẾU DÙNG FILE EXPIRATION DATE THÌ KHÔNG CẦN DÙNG DÒNG IF NÀY
        exp = True
    else:
        exp = False

    if time_now.time() >= datetime.time(14, 50) or trading_day() == False:
        exit()  

    if time_now.strftime('%H:%M:%S') in list_time_5:

        # GET DATA
        data = get_data_ps_adjusted(20)
        data.reset_index(drop=True, inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index("Date")
        #RESAMPLE DATA
        data = data.resample('5Min').agg({'Open': 'first', 'Close': 'last', 'High': 'max', 'Low': 'min', 'Volume': 'sum'}).dropna()
        data = data.reset_index()
        #
        params = {'t1': 0.75, 't2': 13, 't3': 81, 't4': 59, 't5': 37}
        data = alpha(data, params)

        #ALPHA NAME
        name = "MFI"                                                #   CÓ THỂ ĐỂ BÊN NGOÀI VÒNG WHILE NẾU FILE CHỈ CHẠY 1 ALPHA
        #TOKEN CỦA CHAT BOT
        token = '6924463892:AAE05XWY1kuwOZ7u9XfHKEYMTEsf-mskcQY'    #   CÓ THỂ ĐỂ BÊN NGOÀI VÒNG WHILE NẾU FILE CHỈ CHẠY 1 ALPHA
        #ID CỦA GROUP
        id = '-4075432474'                                          #   CÓ THỂ ĐỂ BÊN NGOÀI VÒNG WHILE NẾU FILE CHỈ CHẠY 1 ALPHA
        #TÊN NGƯỜI LÀM ALPHA
        author = "DatNT"

        #ĐƯỜNG DẪN ĐẾN FOLDER CHỨA CP, FILE TRADE
        # path_InPo = f'/home/fin/dat_nt/nas/PS_{author}_{name}.txt'
        # path_CP = f'/home/fin/dat_nt/nas/PS_{author}_{name}_CP.txt'  

        #ĐƯỜNG DẪN ĐẾN FOLDER CHỨA PNL_DAILY, THƯỜNG LÀ THƯ MỤC TRONG Ổ NAS CÁ NHÂN
        path_csv_daily = f'F:/alpha_live_pos/DatNT/PS_{author}_{name}.csv'
        path_csv_intraday = f'F:/alpha_live_pos/DatNT/Intraday/Intraday_{author}_{name}.csv'
        
        df_csv, inputPos, CP = DumpCSV_and_MesToTele(name, path_csv_intraday, data['pos'], data['Close'], token, id, position_input=5, fee=0.8)#position_input: số hợp đồng cần trade
        df_daily = PNL_per_day(path_csv_daily, df_csv['profit_today'])

        path_InPo = f'/home/fin/dat_nt/nas/PS_{author}_{name}.txt'
        path_CP = f'/home/fin/dat_nt/nas/PS_{author}_{name}CP.txt'
        
        position_input(inputPos, path_InPo)
        position_report(CP, path_CP)
        sleep(1)#NẾU CHẠY 1-2 ALPHA THÌ NÊN CHO SLEEP
