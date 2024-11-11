from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import datetime
import optuna
import os
import sys

# set path of this project to sys.path
sys.path.append(os.path.join(os.getcwd(), ".."))
from Baseline.baseline import BaseAlpha, StockDataset, lowpass_filter


class WalkForwardSearcher:
    """
    Implement walk-forward optimization for alpha parameter searching
    """
    def __init__(
        self,
        stock_csv_path: str,
        exp_path: str,
        Alpha: type,
        params: Dict[str, List],
        target_metric: callable,
        train_window: pd.Timedelta = pd.Timedelta(days=365),  # 1 year
        test_window: pd.Timedelta = pd.Timedelta(days=180),   # 6 months
        n_trials: int = 50
    ):
        """
        Args:
            stock_csv_path: Path to stock data CSV
            exp_path: Path to expiration dates CSV 
            Alpha: Alpha class to optimize
            params: Parameter search space dictionary
            target_metric: Function to optimize (takes backtest results and returns score)
            train_window: Training window length
            test_window: Testing window length
            n_trials: Number of optimization trials per window
        """
        self.stock_csv_path = stock_csv_path
        self.exp_path = exp_path
        self.Alpha = Alpha
        self.params = params
        self.target_metric = target_metric
        self.train_window = train_window
        self.test_window = test_window
        self.n_trials = n_trials
        
        # Load full dataset
        self.dataset = StockDataset(
            stock_csv_path=self.stock_csv_path,
            exp_path=self.exp_path,
            train_test_split_time_point=None  # Load full dataset
        )
        self.full_data = self.dataset.get_data()
        self.expiration_date = self.dataset.get_expiration_date()

    def _get_window_data(self, start_date: datetime.datetime, window: pd.Timedelta) -> pd.DataFrame:
        """Get data for a specific time window"""
        end_date = start_date + window
        window_data = self.full_data[
            (self.full_data.index >= start_date) & 
            (self.full_data.index < end_date)
        ]
        return window_data

    def _objective(self, trial, train_data: pd.DataFrame) -> float:
        """Optimization objective for a single window"""
        # Generate parameters
        params = dict()
        for key, value in self.params.items():
            if isinstance(value, list):
                if isinstance(value[0], int):
                    params[key] = trial.suggest_int(key, value[0], value[1])
                elif isinstance(value[0], float):
                    params[key] = trial.suggest_float(key, value[0], value[1])
            else:
                params[key] = value

        # Create model with trial parameters
        model = self.Alpha(
            stock_data=train_data,
            expiration_date=self.expiration_date,
            **params
        )
        
        # Run backtest
        backtest_info = model.backtest(plot=False)
        # If backtest_info is tuple, take first element
        if isinstance(backtest_info, tuple):
            backtest_info = backtest_info[0]
        return self.target_metric(backtest_info)

    def optimize(self) -> Dict[str, Any]:
        """Run walk-forward optimization"""
        results = []
        start_date = self.full_data.index[0]
        end_date = self.full_data.index[-1]
        
        # Iterate through time windows
        current_date = start_date
        while current_date + self.train_window + self.test_window <= end_date:
            print(f"\nOptimizing window starting at {current_date}")
            
            # Get train/test data for current window
            train_data = self._get_window_data(current_date, self.train_window)
            test_data = self._get_window_data(
                current_date + self.train_window, 
                self.test_window
            )
            
            # Create and run optimizer
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self._objective(trial, train_data),
                n_trials=self.n_trials
            )
            
            # Test best parameters on test window
            best_params = study.best_params
            test_model = self.Alpha(
                stock_data=test_data,
                expiration_date=self.expiration_date,
                **best_params
            )
            test_backtest = test_model.backtest(plot=False)
            # If test_backtest is tuple, take first element 
            if isinstance(test_backtest, tuple):
                backtest_info = test_backtest[0]
            else:
                backtest_info = test_backtest
                
            test_score = self.target_metric(backtest_info)
            
            # Store results
            results.append({
                'window_start': current_date,
                'train_end': current_date + self.train_window,
                'test_end': current_date + self.train_window + self.test_window,
                'best_params': best_params,
                'train_score': study.best_value,
                'test_score': test_score,
                'test_metrics': {
                    'profit_after_fee': backtest_info.Profit_after_fee(),
                    'profit_per_day': backtest_info.Profit_per_day(),
                    'sharp_ratio': backtest_info.Sharp_after_fee(),
                    'max_drawdown': backtest_info.MDD()[0],
                    'hit_rate': backtest_info.Hitrate()
                }
            })
            
            # Move to next window
            current_date += self.test_window
            
        return self._analyze_results(results)
    
    def _analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze optimization results across all windows"""
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Calculate parameter stability
        param_stats = defaultdict(dict)
        for param in self.params.keys():
            values = [res['best_params'][param] for res in results]
            param_stats[param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': min(values),
                'max': max(values)
            }
            
        # Calculate performance stability
        performance_stats = {
            'train_score': {
                'mean': df_results['train_score'].mean(),
                'std': df_results['train_score'].std()
            },
            'test_score': {
                'mean': df_results['test_score'].mean(),
                'std': df_results['test_score'].std()
            },
            'profit_after_fee': {
                'mean': df_results['test_metrics'].apply(lambda x: x['profit_after_fee']).mean(),
                'std': df_results['test_metrics'].apply(lambda x: x['profit_after_fee']).std()
            },
            'sharp_ratio': {
                'mean': df_results['test_metrics'].apply(lambda x: x['sharp_ratio']).mean(),
                'std': df_results['test_metrics'].apply(lambda x: x['sharp_ratio']).std()
            }
        }
        
        return {
            'parameter_stability': param_stats,
            'performance_stability': performance_stats,
            'detailed_results': df_results
        }