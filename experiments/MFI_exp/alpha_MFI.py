import pandas as pd
from ta.volume import MFIIndicator
import pandas as pd
import sys
import os

# set path of this project to sys.path
sys.path.append(os.path.join(os.getcwd(), "../../.."))
from Baseline.baseline import BaseAlpha, lowpass_filter


class MFIAlpha(BaseAlpha):
    """MFI-based trading strategy"""
    
    def __init__(self, 
                 stock_data,
                 expiration_date,
                 mfi_period=14,
                 mfi_upper=80,
                 mfi_lower=20,
                 mfi_middle_upper=60,
                 mfi_middle_lower=40,
                 lowpass_filter_ratio=0.75):
        super().__init__(stock_data, expiration_date)
        self.mfi_period = mfi_period
        self.mfi_upper = mfi_upper
        self.mfi_lower = mfi_lower
        self.mfi_middle_upper = mfi_middle_upper
        self.mfi_middle_lower = mfi_middle_lower
        self.lowpass_filter_ratio = lowpass_filter_ratio

    def calculate_indicators(self, df):
        """Calculate MFI indicator"""
        df['Close_filtered'] = pd.Series(lowpass_filter(df['Close'], self.lowpass_filter_ratio))
        
        mfi = MFIIndicator(
            high=df['High'],
            low=df['Low'],
            close=df['Close_filtered'], 
            volume=df['Volume'],
            window=self.mfi_period
        )
        df['mfi'] = mfi.money_flow_index()
        return df

    def generate_signals(self, df, i, position):
        """Generate trading signals based on MFI"""
        if position == 0:
            if df['mfi'].iloc[i] < self.mfi_lower:
                position = -1  # Short when MFI is low
            elif df['mfi'].iloc[i] > self.mfi_upper:
                position = 1   # Long when MFI is high
                
        elif position == -1 and df['mfi'].iloc[i] > self.mfi_middle_upper:
            position = 0  # Exit short
        elif position == 1 and df['mfi'].iloc[i] < self.mfi_middle_lower:
            position = 0  # Exit long
            
        return position