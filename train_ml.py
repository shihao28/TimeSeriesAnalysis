import logging
import pandas as pd
from pmdarima import auto_arima
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import *
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from config import ConfigML
from src.eda import EDA
from src.missingval_analysis import MissingValAnalysis
from src.mlflow_logging import MlflowLogging
from src.feature_select import FeatureSelect

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
        self.metrics = config["evaluation"]["tsa"]
        self.train_data, self.test_data = None, None

    def _preprocessing(self):
        logging.info("Preprocess data...")
        for dt_column, dt_format in self.config['data']['datetime_features'].items():
            self.data[dt_column] = pd.to_datetime(
                self.data[dt_column], format=dt_format)

        self.data.dropna(axis=0, subset=[self.label], inplace=True)

        # self.data['date'].dt.freq
        # self.data['date'].asfreq(self.config['data']['freq'])

        self.data.sort_values(
            by=dt_column, axis=0, ascending=True, inplace=True)

        self.data['Year'] = self.data[dt_column].dt.year
        self.data['Month'] = self.data[dt_column].dt.month
        self.data['Day'] = self.data[dt_column].dt.day
        self.data['Day_of_week'] = self.data[dt_column].dt.dayofweek
        self.data['Week_no'] = self.data[dt_column].dt.week

        # Set datetime column as index
        self.data = self.data.set_index(self.datetime_features_names[0])

        return None

    @staticmethod
    def train_test_split(data, split_ratio):
        logging.info("Train-test splitting...")
        test_start_idx = int(len(data) * split_ratio)
        train_data, test_data = data.iloc[:-test_start_idx,], data.iloc[-test_start_idx:]

        return train_data, test_data

    def _eda(self, data):
        logging.info("Generating EDA report...")
        eda = EDA(
            data, self.label, self.numeric_features_names,
            self.category_features_names).generate_report()

        return None

    def _missing_val_analysis(self, data):
        data = MissingValAnalysis(data).impute_missing_val('rest1')

        return data

    def _select_feat(self, data):
        f_stat_pvalues = FeatureSelect(
            data, self.label, self.numeric_features_names,
            self.category_features_names).select_features()

        return f_stat_pvalues

    def eval(self, y_true, y_pred, fitted_values=None):
        tsa_results = dict()
        tsa_results['mse'] = mean_squared_error(y_true, y_pred)
        tsa_results['rmse'] = tsa_results['mse'] ** 0.5
        tsa_results['mae'] = mean_absolute_error(y_true, y_pred)
        tsa_results['percent_diff'] = np.mean(abs(y_true - y_pred) / y_true) * 100
        tsa_results['r2'] = r2_score(y_true, y_pred)
        tsa_results = pd.DataFrame(tsa_results, index=[0])
        logging.info(f'\n{tsa_results}')

        # Plot predictions against known values
        # https://towardsdatascience.com/time-series-forecasting-using-auto-arima-in-python-bb83e49210cd
        forecast_fig, forecast_ax = plt.subplots()
        forecast_ax.plot(self.data[self.label], label='Actual')
        # forecast_ax = self.data[self.label].plot(legend=True, title=title)
        if fitted_values is not None:
            forecast_ax.plot(fitted_values, label='Fitted Values')
        forecast_ax.plot(y_pred, label='Forecast')
        forecast_ax.set_title(f'Forecast for {self.label}')
        forecast_ax.set(xlabel='', ylabel=self.label)
        forecast_ax.legend()

        return None

    def _mlflow_logging(self, best_train_assets, train_data):
        logging.info("Logging to mlflow...")
        mlflow_logging = MlflowLogging(
            tracking_uri=self.config["mlflow"]["tracking_uri"],
            backend_uri=self.config["mlflow"]["backend_uri"],
            artifact_uri=self.config["mlflow"]["artifact_uri"],
            mlflow_port=self.config["mlflow"]["port"],
            experiment_name=self.config["mlflow"]["experiment_name"],
            run_name=self.config["mlflow"]["run_name"],
            registered_model_name=self.config["mlflow"]["registered_model_name"]
        )
        mlflow_logging.activate_mlflow_server()
        mlflow_logging.logging(
            best_train_assets, train_data, self.label,
            self.split_ratio, self.tune,
            self.config["evaluation"]["classification"])

    def train(self):
        self._preprocessing()

        # Train-test split
        train_data, test_data = self.train_test_split(
            self.data, self.split_ratio)

        # EDA
        self._eda(train_data.copy())

        # Impute missing value
        # self._missing_val_analysis(train_data.copy())

        # Select Features
        self._select_feat(train_data.copy())

        # https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
        model = auto_arima(
            y=train_data[self.label],
            X=train_data[self.config['model']['exog_var']],
            start_p=0, start_q=0,
            max_p=6, max_q=3, m=self.config['model']['seasonal_period'],
            seasonal=self.config['model']['use_seasonal'],
            d=None, trace=True,
            error_action='ignore',   # we don't want to know if an order does not work
            suppress_warnings=True,  # we don't want convergence warnings
            stepwise=True)           # set to stepwise
        # model = ARIMA(train_data[self.label],order=(2, 0, 2)).fit()
        model.summary()

        # Predict
        start = len(train_data)
        end = len(train_data) + len(test_data) - 1
        forecast = model.predict(
            n_periods=len(test_data),
            start=start, end=end, dynamic=False, typ='levels',
            X=test_data[self.config['model']['exog_var']]
            ).rename('Forecast')

        # Evaluate
        self.eval(
            test_data[self.label], forecast,
            model.fittedvalues())

        # logging
        # self.__mlflow_logging()

        return None


if __name__ == "__main__":
    TrainML(ConfigML.train).train()
