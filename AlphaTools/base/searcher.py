from abc import ABC, abstractmethod


class HyperparametersSearcher(ABC):
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def objective(self, trial):
        pass
    
    @abstractmethod
    def optimize_parameters(self, n_trials):
        pass
