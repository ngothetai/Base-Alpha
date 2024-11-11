from typing import Tuple
from ..base.dataset import StockDataset
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


class FutureDatset(StockDataset):
    def __init__(self, stock_csv_path, exp_path, **kwargs) -> None:
        super().__init__(stock_csv_path, exp_path, **kwargs)
        self._split_methods = {
            'sequence': self._split_by_sequence
        }
        
    def _preprocess_data(self) -> None:
        """Prepare data for analysis"""
        self._stock_data['Date'] = pd.to_datetime(self._stock_data['Date'])
        self._stock_data = self._stock_data.set_index('Date')
        self._stock_data = self._stock_data.sort_index()
        self._expiration_date['Date'] = pd.to_datetime(self._expiration_date['Date'])
        self._expiration_date = self._expiration_date.set_index('Date')
        self._expiration_date = self._expiration_date.sort_index()
    
    def _split_by_sequence(self, **kwargs) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame]]:
        try:
            split_date = kwargs['split_date']
        except KeyError:
            raise ValueError("Please provide a split date")
        self._train_data = self._stock_data.loc[self._stock_data.index < split_date]
        self._test_data = self._stock_data.loc[self._stock_data.index >= split_date]
        return tuple([tuple([self._train_data, self._test_data])])
    
    def _split_by_walk_forward(self, **kwargs) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame]]:
        try:
            splitter = TimeSeriesSplit(**kwargs)
            splitter.split(self._stock_data)
            return tuple(splitter)
        except KeyError:
            raise ValueError("Please provide a valid TimeSeriesSplit arguments, follow link: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html")
            
    def _split_train_test_data(self, method: str, **kwargs) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame]]:
        try:
            self._split_methods[method](kwargs)
        except KeyError:
            raise ValueError(f"Method {method} not implemented for splitting data")
