import configparser
import os
import unittest
import pandas as pd
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from src.train import Model

config = configparser.ConfigParser()
config.read("config.ini")


class TestMultiModel(unittest.TestCase):

    def setUp(self) -> None:
        self.multi_model = Model()

    def test_log_reg(self):
        self.assertEqual(self.multi_model.log_reg(), True)


if __name__ == "__main__":
    unittest.main()
