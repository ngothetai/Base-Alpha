import datetime
from typing import Tuple, Dict
from ..utils.F4 import BacktestInformation
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from abc import ABC, abstractmethod


class BaseAlpha(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, stock_data: pd.DataFrame, expiration_date: pd.DataFrame, **kwargs):
        self._data = stock_data
        self._expiration_date = expiration_date
        self._kwargs = kwargs

    @abstractmethod
    def calculate_indicators(self, df) -> pd.DataFrame:
        """Calculate technical indicators needed for the strategy"""
        pass

    @abstractmethod
    def generate_signals(self, df, i, current_position) -> int:
        """Generate trading signals based on indicators"""
        pass

    def __call__(self) -> pd.DataFrame:
        """Main method to calculate positions"""
        df = self._data.copy()
        df = df.reset_index()

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Manage positions
        df['position'] = 0
        current_position = 0
        expiration_date = set(map(lambda x: x.date(), self._expiration_date.index))

        for i in tqdm(range(1, len(df))):
            position = current_position
            
            # Get new position from strategy
            position = self.generate_signals(df, i, position)
                
            # Handle end of day position
            if df['Date'].iloc[i].time() == datetime.time(14, 25):
                if position == -1:
                    position = 0
                    
            # Handle expiration date
            if df['Date'].iloc[i].time() == datetime.time(14, 45):
                if df['Date'].iloc[i].date() in expiration_date:
                    position = 0

            df.iloc[i, df.columns.get_loc('position')] = position
            current_position = position

        return df

    def backtest(self, plot=True) -> BacktestInformation | Tuple[BacktestInformation, Tuple[pd.DataFrame, Dict[str, float]]]:
        """Run backtest on the strategy"""
        df = self()
        backtestInfo = BacktestInformation(df['Date'], df['position'], df['Close'], fee=0.3)
        
        if plot:
            res: Tuple[pd.DataFrame, Dict[str, float]] = backtestInfo.Plot_PNL(plot=True)
            return backtestInfo, res
        return backtestInfo
