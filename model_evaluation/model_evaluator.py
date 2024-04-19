import os
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    matthews_corrcoef,
)
from sklearn.model_selection import cross_val_score
import pandas as pd
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import numpy as np
from utils import save_json
from joblib import load
from pathlib import Path
from config_entity import ModelEvaluationConfig
import matplotlib.pyplot as plt
import seaborn as sns
import google.cloud.aiplatform as aiplatform
from logger import logger


class ModelEvaluation:
    """
    A class for evaluating a machine learning model's performance and deploying it to a model registry.

    Methods:
        prep_data_for_evaluation: Prepares data for model evaluation.
        evaluate_model: Evaluates the model using specified metrics and returns evaluation results.
        plot_confusion_matrix: Plots a confusion matrix for model evaluation.
        register_and_deploy_model: Registers and deploys the model to a model registry.
        evaluate: Executes the evaluation process including data preparation, model evaluation, and deployment.
    """

    def __init__(self, config: ModelEvaluationConfig, experiment_name: str):
        """
        Initializes the ModelEvaluation instance with configuration and experiment name.
        """
        self.config = config
        self.encoder = load(self.config.encoder_path)
        self.model = load(self.config.model_path)
        aiplatform.init(
            experiment="speech-emotion",
            project="firm-site-417617",
            location="us-east1",
            staging_bucket="model-artifact-registry",
        )
        self.experiment = aiplatform.Experiment(experiment_name=experiment_name)

    def prep_data_for_evaluation(self):
        """
        Prepares the data for model evaluation by performing necessary preprocessing steps.

        Returns:
            tuple: A tuple containing the preprocessed test data and corresponding labels.
        """
        test_data = pd.read_parquet(self.config.test_path)
        X_test = test_data.drop("Emotions", axis=1)
        y_test = test_data["Emotions"]
        y_test_enc = self.encoder.transform(np.array(y_test).reshape(-1, 1)).toarray()
        X_test = np.expand_dims(X_test, axis=2)

        return X_test, y_test_enc

    def evaluate_model(self, ytrue, ypred, yproba=None):
        """
        Evaluates the model using specified metrics and returns evaluation results.

        Args:
            ytrue: True labels.
            ypred: Predicted labels.
            yproba: Predicted probabilities (optional).

        Returns:
            tuple: A tuple containing evaluation metrics and confusion matrix.
        """
        metrics = {
            "accuracy": accuracy_score(ytrue, ypred),
            "precision": precision_score(ytrue, ypred, average="macro"),
            "recall": recall_score(ytrue, ypred, average="macro"),
            "f1_score": f1_score(ytrue, ypred, average="macro"),
            "mcc": matthews_corrcoef(ytrue, ypred),
        }
        report = classification_report(ytrue, ypred, output_dict=True)
        conf_matrix = confusion_matrix(ytrue, ypred)
        return metrics, conf_matrix

    def plot_confusion_matrix(self, conf_matrix, class_names):
        """
        Plots a confusion matrix for model evaluation.

        Args:
            conf_matrix: Confusion matrix data.
            class_names: Names of classes for the confusion matrix.
        """
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="g",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        plt.close()
        return plt

    def register_and_deploy_model(self, display_name, version_aliases):
        """
        Registers and deploys the model to a model registry.

        Args:
            display_name: Display name for the model.
            version_aliases: Aliases for the model version.
        """
        TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
        # Register Model in Vertex AI Model Registry
        model = aiplatform.Model.upload(
            display_name=display_name,
            model_id=f"model_{display_name}-{TIMESTAMP}",
            artifact_uri=self.config.model_path,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest",
            is_default_version=True,
            version_aliases=version_aliases,
        )
        endpoint = model.deploy(
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=3,
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
        )

    def evaluate(self):
        """
        Executes the evaluation process including data preparation, model evaluation, and deployment.
        """
        run = self.experiment.start_run(run_name="evaluation_run")
        try:
            X_test, y_test = self.prep_data_for_evaluation()
            y_pred = self.model.predict(X_test)
            metrics, conf_matrix = self.evaluate_model(y_test, y_pred)
            print("Evaluation Metrics:", metrics)
            self.plot_confusion_matrix(conf_matrix, ["Class Names"])

            run.log_metrics(metrics)
            run.log_params({"model_version": "v1", "model_type": "classification"})

            plt_path = "confusion_matrix.png"
            plt.savefig(plt_path)
            run.log_artifact(plt_path)

            model = self.register_and_deploy_model("SER_CNN", ["v1"])

        finally:
            run.update_state(aiplatform.gapic.Execution.State.COMPLETE)
