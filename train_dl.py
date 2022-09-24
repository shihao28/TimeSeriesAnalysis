import os
import torch
import copy
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.svm import *
from sklearn.metrics import *

from config import ConfigDL
from train_ml import TrainML
from src.utils import AverageMeter, accuracy


# Set log level
logging.basicConfig(
    level=logging.DEBUG,
    format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')


class TrainDL(TrainML):
    def __init__(self, config):
        super(TrainDL, self).__init__(config)
        self.device = config["device"]
        self.model_algs = config["model"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.criterion = config["criterion"]
        self.optimizer = config["optimizer"]
        self.lr_scheduler = config["lr_scheduler"]
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.required_numeric_features_names = self.config['model_setting']['exog_var']['numeric']
        self.required_category_features_names = self.config['model_setting']['exog_var']['category']
        self.required_features_names = self.required_numeric_features_names + self.required_category_features_names

    def _create_preprocessing_pipeline(self):
        # # Numeric pipeline
        # # if self.label in self.numeric_features_names:
        # #     self.numeric_features_names.remove(self.label)
        # numeric_pipeline = Pipeline([
        #     ("scaler", StandardScaler())
        #     ])

        # # Category pipeline
        # # if self.label in self.category_features_names:
        # #     self.category_features_names.remove(self.label)
        # # category_pipeline = Pipeline([
        # #     ("encoder", OneHotEncoder(drop="if_binary"))
        # #     ])

        # # Preprocessing pipeline
        # col_transformer = ColumnTransformer([
        #     ("numeric_pipeline", numeric_pipeline, self.required_numeric_features_names + [self.label]),
        #     # ("category_pipeline", category_pipeline, self.category_features_names),
        #     ])
        # preprocessing_pipeline = Pipeline([
        #     ("column_transformer", col_transformer),
        #     # ("outlier", CustomTransformer(IsolationForest(contamination=0.1, n_jobs=-1))),
        #     # ("imputation", KNNImputer(
        #     #     n_neighbors=5,
        #     #     # add_indicator=final_missingness_report.isin(["MCAR/ MNAR"]).any()
        #     #     )),
        #     # ("feature_eng", FeatureEng(
        #     #     features_names=train_data.columns.drop(self.label))),
        #     # ("select_feat", SelectKBest(score_func=f_classif, k=5)),
        #     # ("model", SVC()),
        # ])

        preprocessing_pipeline = StandardScaler()

        return preprocessing_pipeline

    def _split_sequences(self, sequences, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out - 1
            # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y).squeeze()

    def _setting_pytorch_utils(self):
        self._preprocessing()

        # Get only required column
        self.data = self.data[self.required_features_names + [self.label]]

        # Train-test split
        test_start_idx, train_data, test_data = self.train_test_split(
            self.data, self.split_ratio)

        # Create preprocessing pipeline and scale data
        preprocessing_pipeline = self._create_preprocessing_pipeline()
        preprocessing_pipeline.fit(
            train_data[self.required_numeric_features_names + [self.label]]
        )
        scaled_train_data = preprocessing_pipeline.transform(
            train_data[self.required_numeric_features_names + [self.label]]
            )
        scaled_train_label = scaled_train_data[:, -1:]
        scaled_train_data = np.concatenate(
            [scaled_train_data[:, :-1], train_data[self.required_category_features_names],
            scaled_train_label], axis=1
        )
        scaled_test_data = preprocessing_pipeline.transform(
            test_data[self.required_numeric_features_names + [self.label]]
            )
        scaled_test_label = scaled_test_data[:, -1:]
        scaled_test_data = np.concatenate(
            [scaled_test_data[:, :-1], test_data[self.required_category_features_names],
            scaled_test_label], axis=1
        )
        scaled_data = pd.DataFrame(
            np.concatenate([scaled_train_data, scaled_test_data], 0),
            columns=self.required_features_names + [self.label] 
            )

        X, y = self._split_sequences(
            sequences=scaled_data.values,
            n_steps_in=self.config['model_setting']['n_steps_in'],
            n_steps_out=self.config['model_setting']['n_steps_out'],
            )

        # Train-test split
        self.X_train, self.X_test = X[:-test_start_idx], X[-test_start_idx:]
        self.y_train, self.y_test = y[:-test_start_idx], y[-test_start_idx:]

        # Setup pytorch dataloader
        X_train_torch = torch.from_numpy(self.X_train).float()
        X_test_torch = torch.from_numpy(self.X_test).float()
        y_train_torch = torch.from_numpy(self.y_train)
        y_test_torch = torch.from_numpy(self.y_test)
        datasets, dataloaders = dict(), dict()
        for type_, X, y in zip(["train", "val"], [X_train_torch, X_test_torch], [y_train_torch, y_test_torch]):
            datasets[type_] = torch.utils.data.TensorDataset(X, y)
            dataloaders[type_] = torch.utils.data.DataLoader(
                datasets[type_], batch_size=self.batch_size, shuffle=True,
                num_workers=8, drop_last=False)

        return preprocessing_pipeline, dataloaders

    def _train_one_epoch(
        self, dataloader_train, model,
        criterion, optimizer_, device):

        model.train()
        train_epoch_loss = AverageMeter()
        for inputs, labels in dataloader_train:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer_.zero_grad()

            with torch.set_grad_enabled(True):
                logits = model(inputs)
                train_batch_loss = criterion(logits.squeeze(), labels.float())

                train_batch_loss.backward()
                optimizer_.step()

            train_epoch_loss.update(train_batch_loss, inputs.size(0))

        return model, train_epoch_loss.avg

    def _validate(
        self, dataloader_eval, model, criterion,
        device, preprocessing_pipeline, print_tsa_report=False,
        fitted_values=None):

        model.eval()
        val_epoch_loss = AverageMeter()
        labels_all = []
        forecasts_all = []
        for inputs, labels in dataloader_eval:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                logits = model(inputs)
                val_batch_loss = criterion(logits.squeeze(), labels.float())

            labels_all.append(labels)
            forecasts_all.append(logits)

            val_epoch_loss.update(val_batch_loss, inputs.size(0))

        labels_all = torch.cat(labels_all, 0).cpu().numpy()
        forecasts_all = torch.cat(forecasts_all, 0).cpu().numpy()
        if print_tsa_report:
            # Inverse transform
            dummy_numeric_inputs = np.zeros(
                (len(fitted_values), len(self.required_numeric_features_names)))

            numeric_inputs_fitted_values = np.concatenate([
                dummy_numeric_inputs, fitted_values], axis=1)
            fitted_values = preprocessing_pipeline.inverse_transform(
                numeric_inputs_fitted_values)[:, -1:]
            fitted_values = pd.DataFrame(
                fitted_values, index=self.data.index[-len(labels_all)-len(fitted_values):-len(labels_all)])

            dummy_numeric_inputs = np.zeros(
                (len(labels_all), len(self.required_numeric_features_names)))

            numeric_inputs_labels_all = np.concatenate([
                dummy_numeric_inputs, labels_all[:, np.newaxis]], axis=1)
            y_true = preprocessing_pipeline.inverse_transform(
                numeric_inputs_labels_all)[:, -1:]
            y_true = pd.DataFrame(y_true, index=self.data.index[-len(y_true):])

            numeric_inputs_forecasts_all = np.concatenate([
                dummy_numeric_inputs, forecasts_all], axis=1)
            y_pred = preprocessing_pipeline.inverse_transform(
                numeric_inputs_forecasts_all)[:, -1:]
            y_pred = pd.DataFrame(y_pred, index=self.data.index[-len(y_pred):])

            tsa_report = self.eval(
                y_true=y_true, y_pred=y_pred, fitted_values=fitted_values)

        return forecasts_all, val_epoch_loss.avg

    def train(self):
        preprocessing_pipeline, dataloaders = self._setting_pytorch_utils()

        train_assets = dict()
        best_loss = dict()
        for model_alg_name, model in self.model_algs.items():
            logging.info(f"Training {model_alg_name}...")

            model.to(self.device)

            # Initialize optimizer and lr scheduler
            optimizer_ = self.optimizer["alg"](
                params=model.parameters(), **self.optimizer["param"])
            lr_scheduler_ = self.lr_scheduler["alg"](
                optimizer=optimizer_, **self.lr_scheduler["param"])

            best_state_dict = copy.deepcopy(model.state_dict())
            best_model_loss = 1e10
            train_loss, val_loss, lr = [], [], []
            for epoch in range(self.epochs):

                # Train
                model, train_epoch_loss =\
                    self._train_one_epoch(
                        dataloaders['train'], model,
                        self.criterion, optimizer_, self.device)
                train_loss.append(train_epoch_loss.item())
                logging.info(
                    f"Epoch {epoch:3d}/{self.epochs-1:3d} {'Train':5s}, "
                    f"Loss: {train_epoch_loss:.4f}, ")

                # Eval
                _, val_epoch_loss = self._validate(
                    dataloaders['val'], model, self.criterion,
                    self.device, preprocessing_pipeline)
                val_loss.append(val_epoch_loss.item())
                logging.info(
                    f"Epoch {epoch:3d}/{self.epochs-1:3d} {'Val':5s}, "
                    f"Loss: {val_epoch_loss:.4f}, ")

                lr.append(lr_scheduler_.get_last_lr()[0])

                if val_epoch_loss < best_model_loss:
                    best_model_loss = val_epoch_loss
                    best_state_dict = copy.deepcopy(model.state_dict())

                lr_scheduler_.step()

            logging.info('Best Val Loss: {:4f}'.format(best_model_loss))

            # Load best model
            model.load_state_dict(best_state_dict)

            # TSA report
            fitted_values, val_epoch_loss = self._validate(
                dataloaders['train'], model, self.criterion,
                self.device, preprocessing_pipeline)
            _, val_epoch_loss = self._validate(
                dataloaders['val'], model, self.criterion,
                self.device, preprocessing_pipeline, True, fitted_values)

            # Save best model
            torch.save(model.state_dict(), f"{model_alg_name}.pth")

            pd.DataFrame({
                'Epochs': range(self.epochs), 'Learning Rate': lr,
                'Training Loss': train_loss, 'Validation Loss': val_loss,
                }).to_csv(f"{model_alg_name}.csv", index=False)

            logging.info(f"Training {model_alg_name} completed")

        best_loss[model_alg_name] = best_model_loss

        return None


if __name__ == '__main__':
    TrainDL(ConfigDL.train).train()
