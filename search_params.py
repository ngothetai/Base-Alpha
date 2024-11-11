import optuna
import datetime
import sys
import os
# set path of this project to sys.path
sys.path.append(os.path.join(os.getcwd(), ".."))
from Baseline.baseline import BaseAlpha, StockDataset, lowpass_filter
from ta.volume import MFIIndicator
import pandas as pd
from typing import Any, Dict, List


class AlphaParamsSearcher:
    def __init__(self,
                 stock_csv_path,
                 exp_path,
                 Alpha,
                 params: Dict[str, Any],
                 target,
                 directions: List[str] = ["maximize"]
                ):
        self.stock_csv_path = stock_csv_path
        self.exp_path = exp_path
        self.Alpha = Alpha
        self.params = params
        self.target = target
        self.directions = directions
        # Create dataset and split to train and test sets
        self.dataset = StockDataset(
            stock_csv_path=self.stock_csv_path,
            exp_path=self.exp_path,
            train_test_split_time_point=datetime.datetime(2023, 1, 1, 0, 0, 0)  # Split data at 2023-01-01
        )
        self.train_set, self.test_set = self.dataset.get_data()
        self.expiration_date = self.dataset.get_expiration_date()

    def objective(self, trial):
        # Define the parameter search space
        params = dict()
        for key, value in self.params.items():
            if isinstance(value, list):
                if isinstance(value[0], int):
                    params[key] = trial.suggest_int(key, value[0], value[1])
                elif isinstance(value[0], float):
                    params[key] = trial.suggest_float(key, value[0], value[1])
            else:
                params[key] = value

        print("==== Checking: ====", params)
        # Create model instance with trial parameters
        # Searching for the best parameters
        model = self.Alpha(
            stock_data=self.train_set,
            expiration_date=self.expiration_date,
            **params
        )

        print(f"Testing model with parameters on train set...")
        # Run backtest
        backtest_info = model.backtest(plot=True)
        
        objective_value = self.target(backtest_info)
        
        print("Testing model with parameters on test set...")
        
        return objective_value

    def optimize_parameters(self, n_trials=100):
        study = optuna.create_study(directions=self.directions)
        study.optimize(self.objective, n_trials=n_trials)

        print("Best parameters:", study.best_params)
        print("Best value:", study.best_value)

        # Create model with best parameters and show results
        best_model = BaseAlpha(
            stock_csv_path='data/data1mins.csv',
            exp_path='data/expiration_date.csv',
            **study.best_params
        )
        best_backtest = best_model.backtest(plot=True)
        
        # Print detailed metrics
        print("\nDetailed Performance Metrics:")
        print(f"Sharpe Ratio: {best_backtest.Sharp_after_fee():.2f}")
        print(f"Profit after fees: {best_backtest.Profit_after_fee():.2f}")
        print(f"Maximum Drawdown: {best_backtest.MDD()[0]:.2f}")
        print(f"Profit per day: {best_backtest.Profit_per_day():.2f}")
        print(f"Profit per year: {best_backtest.Profit_per_year():.2f}")
        print(f"Hit Rate: {best_backtest.Hitrate():.2f}")

        return study.best_params
