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
        # super(TrainDL, self).__init__(config)
        self.config = config
        self.device = config["device"]
        self.data = pd.read_csv(config["data"]["data_path"])
        self.label = config["data"]["label"]
        self.numeric_features_names = config["data"]["numeric_features"]
        self.category_features_names = config["data"]["category_features"]
        self.datetime_features_names = list(config["data"]["datetime_features"].keys())
        self.model_algs = config["model"]
        self.split_ratio = config["train_val_test_split"]["split_ratio"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.criterion = config["criterion"]
        self.optimizer = config["optimizer"]
        self.lr_scheduler = config["lr_scheduler"]
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None

    def _create_preprocessing_pipeline(self, train_data):
        # Numeric pipeline
        if self.label in self.numeric_features_names:
            self.numeric_features_names.remove(self.label)
        numeric_pipeline = Pipeline([
            ("scaler", StandardScaler())
            ])

        # Category pipeline
        if self.label in self.category_features_names:
            self.category_features_names.remove(self.label)
        category_pipeline = Pipeline([
            ("encoder", OneHotEncoder(drop="if_binary"))
            ])

        # Preprocessing pipeline
        col_transformer = ColumnTransformer([
            ("numeric_pipeline", numeric_pipeline, self.numeric_features_names),
            ("category_pipeline", category_pipeline, self.category_features_names),
            ])
        preprocessing_pipeline = Pipeline([
            ("column_transformer", col_transformer),
            # ("outlier", CustomTransformer(IsolationForest(contamination=0.1, n_jobs=-1))),
            ("imputation", KNNImputer(
                n_neighbors=5,
                # add_indicator=final_missingness_report.isin(["MCAR/ MNAR"]).any()
                )),
            ("feature_eng", FeatureEng(
                features_names=train_data.columns.drop(self.label))),
            # ("select_feat", SelectKBest(score_func=f_classif, k=5)),
            # ("model", SVC()),
        ])

        return preprocessing_pipeline

    def _split_sequences(self, sequences, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix,], sequences[end_ix:out_end_ix, -1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y).squeeze()

    @staticmethod
    def train_test_split(data, split_ratio):
        logging.info("Train-test splitting...")
        test_start_idx = int(len(data) * split_ratio)
        train_data, test_data = data[:-test_start_idx,], data[-test_start_idx:]

        return train_data, test_data

    def _setting_pytorch_utils(self):
        self._preprocessing()

        X, y = self._split_sequences(
            sequences=self.data[self.config['model_setting']['exog_var'] + [self.label]].values,
            n_steps_in=self.config['model_setting']['n_steps_in'],
            n_steps_out=self.config['model_setting']['n_steps_out'])

        # Train-test split
        self.X_train, self.X_test = self.train_test_split(
            X, self.split_ratio)
        self.y_train, self.y_test = self.train_test_split(
            y, self.split_ratio)

        # Create preprocessing and label encoder pipeline
        # preprocessing_pipeline, label_pipeline =\
        #     self._create_preprocessing_pipeline(train_data)
        # preprocessing_pipeline.fit(
        #     train_data.drop(self.label, axis=1),
        #     train_data[self.label])
        # label_pipeline.fit(train_data[self.label])

        # Transform dataset
        # X_train = preprocessing_pipeline.transform(
        #     train_data.drop(self.label, axis=1))
        # X_test = preprocessing_pipeline.transform(
        #     test_data.drop(self.label, axis=1))
        # y_train = label_pipeline.transform(train_data[self.label])
        # y_test = label_pipeline.transform(test_data[self.label])

        # Load dataset onto pytorch loader
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

        return dataloaders

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
            # acc1 = accuracy(logits, labels.data)[0]
            # train_epoch_accuracy.update(acc1.item(), inputs.size(0))

        return model, train_epoch_loss.avg


    def _validate(
        self, dataloader_eval, model, criterion,
        device, print_tsa_report=False):

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
            y_true = pd.DataFrame(
                labels_all, index=self.data.index[-len(self.X_test):])
            y_pred = pd.DataFrame(
                forecasts_all, index=self.data.index[-len(self.X_test):])
            tsa_report = self.eval(
                y_true=y_true, y_pred=y_pred)
            logging.info(f"\n{tsa_report}")

        return val_epoch_loss.avg

    def train(self):
        dataloaders = self._setting_pytorch_utils()

        train_assets = dict()
        best_loss = 1e10
        for model_alg_name, model in self.model_algs.items():
            logging.info(f"Training {model_alg_name}...")

            model.to(self.device)

            # Initialize optimizer and lr scheduler
            optimizer_ = self.optimizer["alg"](
                params=model.parameters(), **self.optimizer["param"])
            lr_scheduler_ = self.lr_scheduler["alg"](
                optimizer=optimizer_, **self.lr_scheduler["param"])

            best_state_dict = copy.deepcopy(model.state_dict())
            best_accuracy = 0
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
                val_epoch_loss = self._validate(
                    dataloaders['val'], model, self.criterion,
                    self.device, False)
                val_loss.append(val_epoch_loss.item())
                logging.info(
                    f"Epoch {epoch:3d}/{self.epochs-1:3d} {'Val':5s}, "
                    f"Loss: {val_epoch_loss:.4f}, ")

                lr.append(lr_scheduler_.get_last_lr()[0])

                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss
                    best_state_dict = copy.deepcopy(model.state_dict())

                lr_scheduler_.step()

            logging.info('Best Val Acc: {:4f}'.format(best_accuracy))

            # Load best model
            model.load_state_dict(best_state_dict)

            # Classification report
            val_epoch_loss = self._validate(
                dataloaders['val'], model, self.criterion,
                self.device, True)

            # Save best model
            torch.save(model.state_dict(), f"{model_alg_name}.pth")

            pd.DataFrame({
                'Epochs': range(self.epochs), 'Learning Rate': lr,
                'Training Loss': train_loss, 'Validation Loss': val_loss,
                }).to_csv(f"{model_alg_name}.csv", index=False)

            logging.info("Training completed")

        return None


if __name__ == '__main__':
    TrainDL(ConfigDL.train).train()
