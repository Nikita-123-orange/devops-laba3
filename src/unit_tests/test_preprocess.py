import configparser
import os
import unittest
import tempfile
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

# Добавляем путь к исходникам
import sys
sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from main import app  # предполагается, что FastAPI приложение создаётся в main.py
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

    @patch('src.api.router.test.Predictor')
    def test_smoke_endpoint(self, MockPredictor):
        mock_instance = MockPredictor.return_value
        mock_instance.smoke_test.return_value = ("LOG_REG", {"accuracy": 0.95})
        response = client.post("/test/smoke")
        assert response.status_code == 200

    def test_functional_endpoint(self):
        """Проверка функционального эндпоинта."""
        response = client.post(self.func_url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["test_type"], "func")
        self.assertIsInstance(data["scores"], dict)
        self.assertIn("accuracy", data["scores"])
        self.assertIsInstance(data["scores"]["accuracy"], float)

    def test_predict_endpoint_with_image(self):
        """Проверка эндпоинта предсказания на примере изображения."""
        # Создаём временное изображение 28x28 (размер MNIST)
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            img = Image.new('L', (28, 28), color=0)
            # Рисуем что-то похожее на цифру
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.rectangle((5, 5, 10, 10), fill=255)
            img.save(tmp.name, format="PNG")
            tmp.seek(0)
            # Отправляем файл
            files = {"file": (tmp.name, open(tmp.name, "rb"), "image/png")}
            response = client.post(self.predict_url, files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("scores", data)
        self.assertIsInstance(data["scores"], dict)
        self.assertIn("confidence", data["scores"])
        self.assertIn("predicted_class", data["scores"])
        self.assertIn("message", data)
        self.assertIsInstance(data["model"], str)

    def test_predict_endpoint_invalid_file(self):
        """Проверка реакции на неподдерживаемый тип файла."""
        files = {"file": ("test.txt", b"some text", "text/plain")}
        response = client.post(self.predict_url, files=files)
        self.assertEqual(response.status_code, 500)  # или 400 в зависимости от реализации
        self.assertIn("detail", response.json())

    def test_predict_endpoint_empty_csv(self):
        """Проверка предсказания с пустым CSV."""
        with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
            tmp.write(b"col1,col2\n")  # только заголовок, нет строк
            tmp.seek(0)
            files = {"file": (tmp.name, open(tmp.name, "rb"), "text/csv")}
            response = client.post(self.predict_url, files=files)
        # Должна быть ошибка, т.к. нет данных
        self.assertEqual(response.status_code, 500)


if __name__ == "__main__":
    unittest.main()