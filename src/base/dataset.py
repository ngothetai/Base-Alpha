from abc import ABC, abstractmethod
from typing import Optional, Tuple
import pandas as pd


class StockDataset(ABC):
    """Split dataset into train and test set by time point"""
    @abstractmethod
    def __init__(self, stock_csv_path: str, exp_path: str, **kwargs) -> None:
        self._stock_data = pd.read_csv(stock_csv_path)
        self._expiration_date = pd.read_csv(exp_path)
        self._preprocess_data()

    @abstractmethod
    def _preprocess_data(self) -> None:
        """Prepare data for analysis"""
        pass
    
    @abstractmethod
    def _split_train_test_data(self, method: Optional[str], **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Define the method to split the time series data into train and test set"""
        pass

    def get_stock_data(self, method: Optional[str], **kwargs) -> pd.DataFrame | tuple[tuple[pd.DataFrame, pd.DataFrame]]:
        if method is None:
            return self._stock_data
        else:
            return self._split_train_test_data(method, **kwargs)

    def get_expiration_date(self) -> pd.DataFrame:
        return self._expiration_date
