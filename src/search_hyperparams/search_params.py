import optuna
from optuna.trial import Trial
import datetime
import sys
import os
# set path of this project to sys.path
sys.path.append(os.path.join(os.getcwd(), ".."))
from ..dataset.stock import StockDataset
from ta.volume import MFIIndicator
import pandas as pd
from typing import Any, Callable, Dict, List, Optional
from ..base.alpha import BaseAlpha
from ..utils.F4 import BacktestInformation


class AlphaParamsSearcher:
    def __init__(self,
                 stock_csv_path: str,
                 exp_path: str,
                 alpha: BaseAlpha,
                 params: Dict[str, Any],
                 target_func: Callable[[BacktestInformation], float],
                 directions: Optional[List[str]] = None,
                 direction: Optional[str] = None
                ):
        self.stock_csv_path = stock_csv_path
        self.exp_path = exp_path
        self.alpha = alpha
        self.params = params
        self.target_func = target_func
        self.directions = directions
        self.direction = direction
        # Create dataset and split to train and test sets
        self.dataset = StockDataset(
            stock_csv_path=self.stock_csv_path,
            exp_path=self.exp_path,
            train_test_split_time_point=datetime.datetime(2023, 1, 1, 0, 0, 0)  # Split data at 2023-01-01
        )
        self.train_set, self.test_set = self.dataset.get_data()
        self.expiration_date = self.dataset.get_expiration_date()
        
    def _convert_params_to_trial(self, trial: Trial) -> Dict[str, Any]:
        params = dict()
        for key, value in self.params.items():
            if isinstance(value, list):
                if isinstance(value[0], int):
                    params[key] = trial.suggest_int(key, value[0], value[1])
                elif isinstance(value[0], float):
                    params[key] = trial.suggest_float(key, value[0], value[1])
            else:
                params[key] = value
        return params

    def objective(self, trial):
        # Define the parameter search space
        params = self._convert_params_to_trial(trial)

        print("==== Checking: ====", params)
        # Create model instance with trial parameters
        # Searching for the best parameters
        model: BaseAlpha = self.alpha(
            stock_data=self.train_set,
            expiration_date=self.expiration_date,
            **params
        )

        print(f"Testing model with parameters on train set...")
        # Run backtest
        
        backtest_info = model.backtest(plot=True)
        
        objective_value = self.target_func(backtest_info)
        
        print("Testing model with parameters on test set...")
        
        return objective_value

    def optimize_parameters(self, n_trials=100):
        if self.direction and self.directions:
            raise ValueError("Please provide either a direction or directions for the study")
        elif self.direction:
            study = optuna.create_study(direction=self.direction)
        elif self.directions:
            study = optuna.create_study(directions=self.directions)
        else:
            raise ValueError("Please provide a direction or directions for the study")
        
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
