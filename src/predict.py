import configparser
from datetime import datetime
import os
import json
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import traceback
from src.logger import Logger

SHOW_LOG = True


class Predictor():

    def __init__(self, model: str = "LOG_REG", test_type: str = "smoke") -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        self.model_name = model
        self.root_dir = os.getcwd()

        # --- загрузка модели ---
        model_rel_path = self.config[model]["path"].replace('\\', '/')
        model_abs_path = os.path.join(self.root_dir, model_rel_path)
        try:
            with open(model_abs_path, "rb") as f:
                self.model = pickle.load(f)
        except FileNotFoundError as e:
            self.log.error(traceback.format_exc())
            raise RuntimeError(f"Model file not found: {model_abs_path}") from e

        self.test_type = test_type

        # --- загрузка тестовых данных ---
        x_test_path = os.path.join(self.root_dir, self.config["SPLIT_DATA"]["x_test"])
        y_test_path = os.path.join(self.root_dir, self.config["SPLIT_DATA"]["y_test"])
        self.X_test = pd.read_csv(x_test_path, index_col=0)
        self.y_test = pd.read_csv(y_test_path, index_col=0)

        # --- загрузка обученного StandardScaler (вместо создания нового) ---
        try:
            scaler_rel_path = self.config["SCALER"]["path"].replace('\\', '/')
            scaler_abs_path = os.path.join(self.root_dir, scaler_rel_path)
            with open(scaler_abs_path, "rb") as f:
                self.sc = pickle.load(f)
            self.log.info(f"StandardScaler loaded from {scaler_abs_path}")
        except (KeyError, FileNotFoundError) as e:
            self.log.error("SCALER section not found in config.ini or scaler file missing. Run train.py first.")
            raise RuntimeError("Missing fitted StandardScaler") from e

        # --- масштабирование тестовых данных ---
        self.X_test = self.sc.transform(self.X_test)

        self.log.info("Predictor is ready")

    def metrics_calculation(self, y_true, y_pred) -> dict[str, float]:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        return {
            "accuracy": float(acc),
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall)
        }

    def smoke_test(self) -> tuple[str, dict]:
        """Smoke test: быстрая проверка на тестовых данных."""
        try:
            self.log.info(f"Начало smoke теста для модели {self.model_name}")
            y_pred = self.model.predict(self.X_test)
            y_true = self.y_test.values.ravel()
            scores = self.metrics_calculation(y_true, y_pred)
            self.log.info(f'Модель {self.model_name} прошла smoke тест. Scores: {scores}')
            return self.model_name, scores
        except Exception as e:
            self.log.error(traceback.format_exc())
            raise RuntimeError(f"Smoke test failed: {e}") from e

    def functional_test(self) -> tuple[str, dict]:
        """Functional test: тестирование на всех JSON файлах из директории tests."""
        try:
            self.log.info(f"Начало functional теста для модели {self.model_name}")
            tests_path = os.path.join(self.root_dir, "tests")
            if not os.path.exists(tests_path):
                raise RuntimeError(f"Tests directory not found: {tests_path}")

            test_files = [f for f in os.listdir(tests_path) if f.endswith('.json')]
            if not test_files:
                raise RuntimeError("No JSON test files found in tests directory")

            self.log.info(f"Найдено {len(test_files)} JSON файлов для тестирования")
            y_true_array = []
            y_pred_array = []

            for test_file in sorted(test_files):
                test_file_path = os.path.join(tests_path, test_file)
                try:
                    with open(test_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        X = self.sc.transform(pd.json_normalize(data, record_path=['X']).values)
                        y_true = pd.json_normalize(data, record_path=['y']).iloc[:, 0].astype(int).values
                        y_pred = self.model.predict(X)
                        y_true_array.extend(y_true)
                        y_pred_array.extend(y_pred)
                        self.log.info(f"  Обработан файл: {test_file}")
                except (json.JSONDecodeError, KeyError) as e:
                    self.log.warning(f"Пропущен файл {test_file}: {str(e)}")
                    continue
                except Exception as e:
                    self.log.error(f"Ошибка при обработке {test_file}: {traceback.format_exc()}")
                    raise RuntimeError(f"Functional test failed for {test_file}: {e}") from e

            if not y_true_array:
                raise RuntimeError("No valid test data processed")

            scores = self.metrics_calculation(y_true_array, y_pred_array)
            self.log.info(f'Модель {self.model_name} прошла Functional тест. Scores: {scores}')
            return self.model_name, scores
        except Exception as e:
            self.log.error(traceback.format_exc())
            raise RuntimeError(f"Functional test failed: {e}") from e

    def predict_from_features(self, features: np.ndarray) -> tuple[int, float]:
        """Предсказание класса на основе признаков."""
        try:
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            if features.shape[1] != 784:
                self.log.warning(f"Feature count mismatch: got {features.shape[1]}, expected 784")
            features_scaled = self.sc.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            confidence = 0.0
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features_scaled)
                confidence = float(np.max(proba))
            self.log.info(f"Предсказание: класс {prediction}, confidence: {confidence:.4f}")
            return int(prediction), confidence
        except Exception as e:
            self.log.error(traceback.format_exc())
            raise RuntimeError(f"Prediction failed: {e}") from e


if __name__ == "__main__":
    predictor = Predictor()
    predictor.smoke_test()