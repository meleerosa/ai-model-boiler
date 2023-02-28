import logging
import os
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from modules.common.utils.decorator import TryDecorator
from modules.config import Config

preprocess_logger = logging.getLogger("Preprocess")


class Preprocess:
    """ Preprocess class
    
    Attributes:
        _raw_data (pd.DataFrame): raw data를 저장
        _preprocessed_data (pd.DataFrame): 전처리된 데이터를 저장

    """

    _raw_data: pd.DataFrame
    _preprocessed_data: pd.DataFrame
    _train_input: np.ndarray
    _train_target: np.ndarray
    _test_input: np.ndarray
    _test_target: np.ndarray

    @property
    def raw_data(self):
        return self._raw_data

    @raw_data.setter
    def raw_data(self, value):
        self._raw_data = value

    def __init__(self) -> None:
        self._config = Config.instance().config

    def load_data(self):
        self._raw_data = pd.read_csv(
            os.path.join(self._config["path"]["data"], "data.csv")
        )
        pass

    def preprocess(self) -> None:
        self._preprocessed_data = self._raw_data
        preprocess_logger.info("Model Data Count-------------------------------------")
        preprocess_logger.info("raw dataset        : " + str(len(self._raw_data)))
        preprocess_logger.info(
            "preprocessed dataset  : " + str(len(self._preprocessed_data))
        )
        preprocess_logger.info("-----------------------------------------------------")
        pass

    def run(self) -> Any:
        self.load_data()
        self.preprocess()
        return self._preprocessed_data
