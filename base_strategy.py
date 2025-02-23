# strategies/base_strategy.py

import logging
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def get_signal(self, df, instrument):
        pass

    def _get_price_precision(self, instrument):
        return 3 if 'JPY' in instrument else 5

    def round_price(self, price, instrument):
        precision = self._get_price_precision(instrument)
        return round(price, precision)