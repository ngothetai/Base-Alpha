import datetime
from Baseline.utils.F4 import BacktestInformation
import pandas as pd
from scipy.signal import butter,lfilter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from abc import ABC, abstractmethod


def lowpass_filter(signal, ratio):
    b, a = butter(1, ratio, btype='low', analog=False)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal


class StockDataset:
    "Split dataset into train and test set by time point"
    def __init__(self, stock_csv_path: str, exp_path: str, train_test_split_time_point: datetime.timedelta | None):
        self._data = pd.read_csv(stock_csv_path)
        self._expiration_date = pd.read_csv(exp_path)
        self._train_test_split_time_point = train_test_split_time_point
        self._preprocess_data()

    def _preprocess_data(self):
        """Prepare data for analysis"""
        self._data['Date'] = pd.to_datetime(self._data['Date'])
        self._data = self._data.dropna()
        self._data = self._data.set_index('Date')
        self._data = self._data.sort_index()
        self._expiration_date['Date'] = pd.to_datetime(self._expiration_date['Date'])
        if self._train_test_split_time_point:
            self._split_data_by_sequence()
            
    def _split_data_by_sequence(self):
        split_date = self._train_test_split_time_point
        self._train_data = self._data.loc[self._data.index < split_date]
        self._test_data = self._data.loc[self._data.index >= split_date]
        self._data = tuple([self._train_data, self._test_data])

    def get_data(self) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        return self._data

    def get_expiration_date(self) -> pd.DataFrame:
        return self._expiration_date


class BaseAlpha(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, stock_data: pd.DataFrame, expiration_date: pd.DataFrame):
        self._data = stock_data
        self._expiration_date = expiration_date 

    @abstractmethod
    def calculate_indicators(self, df):
        """Calculate technical indicators needed for the strategy"""
        pass

    @abstractmethod
    def generate_signals(self, df, i, current_position):
        """Generate trading signals based on indicators"""
        pass

    def calculate_signals(self):
        """Main method to calculate positions"""
        df = self._data.copy()
        df = df.reset_index()

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Manage positions
        df['position'] = 0
        current_position = 0
        expiration_date = set(map(lambda x: x.date(), self._expiration_date['Date']))

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

    def backtest(self, plot=True):
        """Run backtest on the strategy"""
        df = self.calculate_signals()
        backtestInfo = BacktestInformation(df['Date'], df['position'], df['Close'], fee=0.3)
        
        if plot:
            res = backtestInfo.Plot_PNL()
            return backtestInfo, res
        return backtestInfo
