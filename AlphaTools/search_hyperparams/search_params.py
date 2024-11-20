import optuna
from optuna.trial import Trial
from ..dataset.stock import FutureDataset
from typing import Any, Callable, Dict, List
from ..base.alpha import BaseAlpha
from ..utils.F4 import BacktestInformation


class AlphaParamsSearcher:
    def __init__(self,
        stock_csv_path: str,
        exp_path: str,
        alpha: BaseAlpha.__class__,
        params: Dict[str, Any],
        target_func: Callable[[BacktestInformation], float],
        directions: List[str] | str,
        method: str,
        **kwargs
    ):
        self.stock_csv_path = stock_csv_path
        self.exp_path = exp_path
        self.alpha = alpha
        self.params = params
        self.target_func = target_func
        self.directions = directions
        self._kwargs = kwargs
        self.method = method

        # Create dataset and split to train and test sets
        self.dataset = FutureDataset(
            stock_csv_path=self.stock_csv_path,
            exp_path=self.exp_path
        )

        try:
            self.train_set, self.test_set = self.dataset.get_stock_data(
                method,
                **kwargs
            )[0]
            #@TODO: Implement with walk_forward dataset (multiple train-test sets)
        except KeyError:
            raise ValueError("Please provide a method to split the dataset and its parameters")
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

    def _objective(self, trial: Trial) -> float:
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
        if isinstance(self.directions, str):
            study = optuna.create_study(direction=self.directions)
        elif isinstance(self.directions, list):
            study = optuna.create_study(directions=self.directions)
        else:
            raise ValueError("Please provide a direction or directions for the study")
        
        study.optimize(self._objective, n_trials=n_trials)
        print("Best parameters:", study.best_params)
        print("Best value:", study.best_value)

        # Create model with the best parameters and show results
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
