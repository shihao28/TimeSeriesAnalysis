import os
import pandas as pd
from subprocess import Popen, DEVNULL
import mlflow
from mlflow.models.signature import infer_signature
from pathlib import Path
import shutil
import requests
import time
import logging


class MlflowLogging:
    def __init__(
        self, tracking_uri, backend_uri, artifact_uri, mlflow_port,
            experiment_name=None, run_name=None, registered_model_name=None):

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.registered_model_name = registered_model_name

        env = {
            "MLFLOW_TRACKING_URI": f"{tracking_uri}:{mlflow_port}",
            "BACKEND_URI": backend_uri,
            "ARTIFACT_URI": artifact_uri,
            "MLFLOW_PORT": mlflow_port
            }
        os.environ.update(env)
        self.cmd_mlflow_server = (
            f"mlflow server --backend-store-uri {backend_uri} "
            f"--default-artifact-root {artifact_uri} "
            f"--host 0.0.0.0 -p {mlflow_port}")

    def activate_mlflow_server(self):
        with open("stderr.txt", mode="wb") as out, open("stdout.txt", mode="wb") as err:
            Popen(self.cmd_mlflow_server, stdout=out, stderr=err, stdin=DEVNULL,
                  universal_newlines=True, encoding="utf-8",
                  env=os.environ, shell=True)

        # Keep pinging until mlfow server is up
        while True:
            try:
                response = requests.get(f'{os.getenv("MLFLOW_TRACKING_URI")}/api/2.0/mlflow/experiments/list')
                if str(response) == "<Response [200]>":
                    logging.warning(f'MLFLOW tracking server response: {str(response)}')
                    break
            except requests.exceptions.ConnectionError:
                logging.warning(f'Tracking server {os.getenv("MLFLOW_TRACKING_URI")} is not up and running')
                time.sleep(1)

    def logging(self, best_train_assets, train_data, label, split_ratio, tune, eval_metrics):
        # check f1 score with cls report
        best_train_pipeline = best_train_assets.get("train_pipeline")
        best_evaluation_results = best_train_assets.get("evaluation_results")
        best_cls_report = self.__get_mlflow_cls_report(best_evaluation_results)
        best_threshold = best_train_assets.get("best_threshold")

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=self.run_name):
            mlflow.log_param('target_variable', label)
            mlflow.log_param('split_ratio', split_ratio)
            mlflow.log_param('tune', tune)
            mlflow.log_param('eval_metrics', eval_metrics)

            mlflow.log_metrics(best_cls_report)
            if best_threshold is not None:
                mlflow.log_metrics("best_threshold", best_threshold)

            signature = infer_signature(
                train_data,
                pd.DataFrame({label: best_train_pipeline.predict(
                    train_data.drop(label, axis=1))}))
            mlflow.sklearn.log_model(
                sk_model=best_train_pipeline, artifact_path="sk_models",
                signature=signature, input_example=train_data.sample(5),
                registered_model_name=self.registered_model_name
                )

            # Store plots as artifacts
            artifact_folder = Path("mlflow_tmp")
            artifact_folder.mkdir(parents=True, exist_ok=True)

            # Storing only figures, pd.DataFrames are excluded
            conf_matrix_fig = best_evaluation_results.get("conf_matrix_fig")
            conf_matrix_fig.savefig(Path(artifact_folder, "conf_matrix.png"))
            fig_all = best_evaluation_results.get("fig")
            for class_label, fig in fig_all.items():
               fig.savefig(Path(artifact_folder, f"fig_{class_label}.png"))
            mlflow.log_artifacts(
                artifact_folder, artifact_path="evaluation_artifacts")
            shutil.rmtree(artifact_folder)
