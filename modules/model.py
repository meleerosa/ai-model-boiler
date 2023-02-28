import logging
import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from modules.common.utils.decorator import TryDecorator
from modules.common.utils.file_handler import chk_and_make_dir
from modules.config import Config

model_logger = logging.getLogger("model")


class Model:
    _input_data: pd.DataFrame
    _train_input: np.ndarray
    _train_target: np.ndarray
    _test_input: np.ndarray
    _test_target: np.ndarray
    _model: keras.models.Model
    _config: dict
    _model_path: str

    @property
    def h_param(self):
        return self._h_param

    def __init__(self, preprocessed_data, h_param: Dict = None) -> None:
        tf.random.set_seed(17)
        self._config = Config.instance().config
        self._model_logger = logging.getLogger("Model")
        self._preprocessed_data = preprocessed_data
        if h_param is None:
            self._h_param = self._config["hyper_parameter"]
        else:
            self._h_param = h_param

    @TryDecorator(logger=model_logger)
    def _split_data(self) -> None:
        if self._preprocessed_data is None or self._preprocessed_data.empty:
            raise Exception("preprocessed_data Empty")

        train_set, test_set = train_test_split(
            self._preprocessed_data, test_size=0.2, random_state=17, shuffle=False
        )

        self._train_input = train_set.drop("target", axis=1).values
        self._train_target = train_set["target"].values
        self._test_input = test_set.drop("target", axis=1).values
        self._test_target = test_set["target"].values

        model_logger.info("Model Data Count-------------------------------------")
        model_logger.info(
            "preprocessed dataset  : " + str(len(self._preprocessed_data))
        )
        model_logger.info("train dataset      : " + str(len(self._train_input)))
        model_logger.info("test dataset       : " + str(len(self._test_input)))
        model_logger.info("-----------------------------------------------------")

    def _build_model(self) -> None:
        self._model = keras.Sequential(
            [
                layers.Dense(
                    self._h_param["dense1_units"],
                    input_dim=self._train_input.shape[1],
                    activation="relu",
                ),
                layers.Dense(1),
            ]
        )

    def _compile_model(self) -> None:
        optimizers = keras.optimizers.Adam(learning_rate=self._h_param["lr"])
        self._model.compile(optimizer=optimizers, loss="mse", metrics=["mape"])

    def _fit(self) -> None:
        es = EarlyStopping(
            monitor="loss",
            mode="min",
            verbose=1,
            patience=self._h_param["early_stopping_rounds"],
            restore_best_weights=True,
        )
        self._model.fit(
            x=self._train_input,
            y=self._train_target,
            batch_size=self._h_param["batch_size"],
            epochs=self._h_param["epochs"],
            callbacks=[es],
            verbose=1,
        )

    def evaluate_model(self) -> float:
        evaluate = self._model.evaluate(
            x=self._test_input, y=self._test_target, return_dict=True
        )
        return evaluate

    def predict(self, input_data: Any) -> Any:
        pred = []
        for each_input in input_data:
            pred.append(self._model.predict(each_input))
        return pred

    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            model_logger.error("Model does not exsist.")
        else:
            self._model = keras.models.load_model(model_path)

    def _save_model(self, model_path: str) -> None:
        chk_and_make_dir(model_path)
        self._model.save(model_path)

    def save_model(self, model_path: str = None) -> None:
        current_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._model_path = model_path
        if self._model_path is None:
            self._model_path = os.path.join(
                self._config["path"]["output"], current_dt, "model"
            )
        self._save_model(self._model_path)

    def load_model(self, model_dt: str) -> None:
        model_path = os.path.join(self._config["path"]["output"], model_dt, "model")
        self._load_model(model_path)

    def fit_and_evaluate(self) -> Dict:
        self._split_data()
        self._build_model()
        self._compile_model()
        self._fit()
        return self.evaluate_model()

    def fit_and_predict_test(self) -> Any:
        self._split_data()
        self._build_model()
        self._compile_model()
        self._fit()
        self.save_model()
        self.evaluate_model()
        return self.predict(self._test_input)
