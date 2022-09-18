import logging
import pandas as pd

from config import ConfigML
from src.eda import EDA
from src.missingval_analysis import MissingValAnalysis

# perform aggregation

# Set log level
logging.basicConfig(
    level=logging.INFO,
    format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')


class TrainML:
    def __init__(self, config):
        logging.info("Initializing...")
        self.config = config
        self.data = pd.read_csv(config["data"]["data_path"])
        self.label = config["data"]["label"]
        self.numeric_features_names = config["data"]["numeric_features"]
        self.category_features_names = config["data"]["category_features"]
        self.datetime_features_names = list(config["data"]["datetime_features"].keys())
        self.split_ratio = config["train_val_test_split"]["split_ratio"]
        self.metrics = config["evaluation"]["regression"]
        self.train_data, self.test_data = None, None

    def __preprocessing(self):
        logging.info("Preprocess data...")
        for dt_column, dt_format in self.config['data']['datetime_features'].items():
            self.data[dt_column] = pd.to_datetime(
                self.data[dt_column], format=dt_format)

        self.data.sort_values(
            by=dt_column, axis=0, ascending=True, inplace=True)

        self.data['Year'] = self.data[dt_column].dt.year
        self.data['Month'] = self.data[dt_column].dt.month
        self.data['Day'] = self.data[dt_column].dt.day
        self.data['Day_of_week'] = self.data[dt_column].dt.dayofweek
        self.data['Week_no'] = self.data[dt_column].dt.week

    @staticmethod
    def train_test_split(data, split_ratio):
        logging.info("Train-test splitting...")
        test_start_idx = int(len(data) * split_ratio)
        train_data, test_data = data.iloc[:-test_start_idx,], data.iloc[-test_start_idx:]

        return train_data, test_data

    def __eda(self, data):
        logging.info("Generating EDA report...")
        eda = EDA(
            data, self.label, self.numeric_features_names,
            self.category_features_names, self.datetime_features_names
            ).generate_report()

    def __missing_val_analysis(self, data):
        data = MissingValAnalysis(data).impute_missing_val('rest1')

        return data

    def __eval(self, train_pipeline, test_data, metrics):
        pass

    def __mlflow_logging(self, best_train_assets, train_data):
        pass

    def train(self):
        self.__preprocessing()

        # Train-test split
        train_data, test_data = self.train_test_split(
            self.data, self.split_ratio)

        # EDA
        self.__eda(train_data.copy())

        self.__missing_val_analysis(train_data.copy())


if __name__ == "__main__":
    TrainML(ConfigML.train).train()
