import configparser
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import traceback
import zipfile
from src.logger import Logger

TEST_SIZE = 0.2
SHOW_LOG = True


class DataMaker():
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            logger = Logger(SHOW_LOG)
            self.config = configparser.ConfigParser()
            self.log = logger.get_logger(__name__)
            self.project_path = os.path.join(os.getcwd(), "data")
            self.root_dir = os.path.dirname(self.project_path)   # корень проекта (папка devops-laba1)
            self.zip_path = os.path.join(self.project_path, "fashion_mnist.zip")
            self._check_and_extract_zip()

            self.data_path = os.path.join(self.project_path, "fashion_mnist.csv")
            self.X_path = os.path.join(self.project_path, "fashion_mnist_x.csv")
            self.y_path = os.path.join(self.project_path, "fashion_mnist_y.csv")
            self.train_path = [os.path.join(self.project_path, "train_fashion_mnist_x.csv"),
                               os.path.join(self.project_path, "train_fashion_mnist_y.csv")]
            self.test_path = [os.path.join(self.project_path, "test_fashion_mnist_x.csv"),
                              os.path.join(self.project_path, "test_fashion_mnist_y.csv")]
            self.log.info(f"{self.__class__.__name__} is ready")
            self._initialized = True

    def _check_and_extract_zip(self):
        if os.path.exists(self.zip_path):
            self.log.info(f"Найден архив {self.zip_path}, начинаю распаковку...")
            try:
                with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                    original_name = zip_ref.namelist()[0]
                    zip_ref.extract(original_name, self.project_path)
                    os.rename(
                        os.path.join(self.project_path, original_name),
                        os.path.join(self.project_path, 'fashion_mnist.csv')
                    )
                self.log.info("Архив успешно распакован")
            except Exception as e:
                self.log.error(f"Ошибка при распаковке архива: {e}")
        else:
            self.log.warning(f"Не найден ZIP архив в {self.project_path}")

    def get_data(self) -> bool:
        dataset = pd.read_csv(self.data_path)
        X = pd.DataFrame(dataset.iloc[:, 1:].values)
        y = pd.DataFrame(dataset.iloc[:, :1].values)
        X.to_csv(self.X_path, index=False)
        y.to_csv(self.y_path, index=False)
        if os.path.isfile(self.X_path) and os.path.isfile(self.y_path):
            self.log.info("X and y data is ready")
            # записываем относительные пути в конфиг
            self.config["DATA"] = {
                'x_data': os.path.relpath(self.X_path, start=self.root_dir).replace('\\', '/'),
                'y_data': os.path.relpath(self.y_path, start=self.root_dir).replace('\\', '/')
            }
            return True
        else:
            self.log.error("X and y data is not ready")
            return False

    def split_data(self, test_size=TEST_SIZE) -> bool:
        self.log.info(self.project_path)
        self.get_data()
        try:
            X = pd.read_csv(self.X_path)
            y = pd.read_csv(self.y_path)
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0)
        self.save_splitted_data(X_train, self.train_path[0])
        self.save_splitted_data(y_train, self.train_path[1])
        self.save_splitted_data(X_test, self.test_path[0])
        self.save_splitted_data(y_test, self.test_path[1])

        # записываем относительные пути
        self.config["SPLIT_DATA"] = {
            'x_train': os.path.relpath(self.train_path[0], start=self.root_dir),
            'y_train': os.path.relpath(self.train_path[1], start=self.root_dir),
            'x_test': os.path.relpath(self.test_path[0], start=self.root_dir),
            'y_test': os.path.relpath(self.test_path[1], start=self.root_dir)
        }
        self.log.info("Train and test data is ready")
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        return (os.path.isfile(self.train_path[0]) and
                os.path.isfile(self.train_path[1]) and
                os.path.isfile(self.test_path[0]) and
                os.path.isfile(self.test_path[1]))

    def save_splitted_data(self, df: pd.DataFrame, path: str) -> bool:
        df = df.reset_index(drop=True)
        df.to_csv(path, index=True)
        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


if __name__ == "__main__":
    data_maker = DataMaker()
    data_maker.split_data()