import configparser
import unittest
import tempfile
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
import sys
import os

# Добавляем путь к исходникам (если нужно)
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from main import app
from src.logger import Logger

SHOW_LOG = True
logger = Logger(SHOW_LOG).get_logger(__name__)

config = configparser.ConfigParser()
config.read("config.ini")

client = TestClient(app)


class TestEndpoints(unittest.TestCase):

    def setUp(self) -> None:
        """Подготовка перед каждым тестом."""
        self.smoke_url = "/test/smoke"
        self.func_url = "/test/func"
        self.predict_url = "/test/predict"
        # Пути к реальным тестовым файлам (если существуют)
        self.csv_path = "tests/sample_test_2.csv"
        self.image_path = "tests/sample_image_0.png"

    def test_smoke(self):
        """Проверка дымового теста модели."""
        response = client.post(self.smoke_url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["model"], "LOG_REG")
        self.assertEqual(data["test_type"], "smoke")
        self.assertIn("accuracy", data["scores"])

    def test_functional(self):
        """Проверка функционального теста модели."""
        response = client.post(self.func_url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["model"], "LOG_REG")
        self.assertEqual(data["test_type"], "func")
        self.assertIn("accuracy", data["scores"])

    def test_predict_with_csv(self):
        """Предсказание по CSV-файлу."""
        # Если реальный CSV существует — используем его
        if os.path.exists(self.csv_path):
            with open(self.csv_path, "rb") as f:
                content = f.read()
        else:
            # Создаём минимальный валидный CSV (2 строки, 784 признака)
            import pandas as pd
            df = pd.DataFrame(np.random.randint(0, 255, size=(2, 784)))
            content = df.to_csv(index=False, header=False).encode('utf-8')

        files = {"file": ("test.csv", content, "text/csv")}
        response = client.post(self.predict_url, files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["model"], "LOG_REG")
        self.assertIn("predicted_class", data["scores"])
        self.assertIn("confidence", data["scores"])
        self.assertIsInstance(data["scores"]["predicted_class"], int)

    def test_predict_with_image(self):
        """Предсказание по grayscale-изображению 28x28."""
        # Если реальное изображение существует — используем его
        if os.path.exists(self.image_path):
            with open(self.image_path, "rb") as f:
                content = f.read()
        else:
            # Создаём тестовое изображение 28x28
            img = Image.new("L", (28, 28), color=128)
            img_bytes = BytesIO()
            img.save(img_bytes, format="PNG")
            content = img_bytes.getvalue()

        files = {"file": ("test_image.png", content, "image/png")}
        response = client.post(self.predict_url, files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["model"], "LOG_REG")
        self.assertIn("predicted_class", data["scores"])
        self.assertIn("confidence", data["scores"])

    def test_predict_invalid_file(self):
        """Передача неподдерживаемого формата файла."""
        files = {"file": ("test.txt", b"just text", "text/plain")}
        response = client.post(self.predict_url, files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("неподдерживаемый тип", response.json()["detail"].lower())

    def test_predict_empty_csv(self):
        """Пустой CSV-файл."""
        empty_csv = BytesIO(b"")
        files = {"file": ("empty.csv", empty_csv, "text/csv")}
        response = client.post(self.predict_url, files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("пуст", response.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()