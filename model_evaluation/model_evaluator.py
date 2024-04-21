import os
from datetime import datetime
import uuid
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
import pandas as pd
import numpy as np
from joblib import load
from keras.models import load_model
from keras.utils import to_categorical
from pathlib import Path
from config_entity import ModelEvaluationConfig
import matplotlib.pyplot as plt
import seaborn as sns
import google.cloud.aiplatform as aiplatform
from logger import logger
import vertexai
from google.oauth2 import service_account


class ModelEvaluation:
    """
    A class for evaluating a machine learning model's performance and deploying it to a model registry.

    Methods:
        prep_data_for_evaluation: Prepares data for model evaluation.
        evaluate_model: Evaluates the model using specified metrics and returns evaluation results.
        plot_confusion_matrix: Plots a confusion matrix for model evaluation.
        register_model: Registers the model to a model registry.
        evaluate: Executes the evaluation process including data preparation, model evaluation, and deployment.
    """

    def __init__(self, config: ModelEvaluationConfig, experiment_name: str='speech-emotion-evaluation'):
        """
        Initializes the ModelEvaluation instance with configuration and experiment name.
        """
        self.config = config
        try:
            self.encoder = load(self.config.encoder_path)
            logger.info("Encoder has been imported successfully")
        except Exception as e:
            logger.error(f"Error loading Encoder: {str(e)}")
        try:
            self.model = load_model(self.config.model_path)  # load keras model
            logger.info(
                f"Model has been successfully loaded from {self.config.model_path}"
            )
        except Exception as e:
            logger.error(f"Error loading Model: {str(e)}")
            raise
        self.credentials = self.authenticate()
        try:
            aiplatform.init(
                experiment=os.getenv("AIPLATFORM_EXPERIMENT", experiment_name),
                project=os.getenv("GOOGLE_CLOUD_PROJECT", "firm-site-417617"),
                location=os.getenv("AIPLATFORM_LOCATION", "us-east1"),
                staging_bucket=os.getenv(
                    "AIPLATFORM_BUCKET", "model-artifact-registry"
                ),
                credentials=self.credentials,
            )
            logger.info("AI Platform initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize AI Platform: {str(e)}")
            raise
        # self.experiment = aiplatform.Experiment(experiment_name=experiment_name)

    def prep_data_for_evaluation(self):
        """
        Prepares the data for model evaluation by performing necessary preprocessing steps.

        Returns:
            tuple: A tuple containing the preprocessed test data and corresponding labels.
        """
        test_data = pd.read_parquet(self.config.test_path)
        X_test = test_data.drop("Emotions", axis=1)
        if X_test.ndim == 2:
            X_test = np.expand_dims(X_test, axis=2)
        y_test = test_data["Emotions"]
        y_test_enc = self.encoder.transform(np.array(y_test).reshape(-1, 1)).toarray()
        y_test_labels = self.encoder.inverse_transform(y_test_enc)
        return X_test, y_test_enc, y_test_labels

    def evaluate_model(self, ytrue, ypred, y_true_label, y_pred_label, yproba=None):
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
            "mcc": matthews_corrcoef(y_true_label, y_pred_label),
        }
        report = classification_report(ytrue, ypred, output_dict=True)
        conf_matrix = confusion_matrix(y_true_label, y_pred_label)
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
        return plt

    def register_model(self, display_name, version_aliases):
        """
        Registers the model to a model registry.

        Args:
            display_name: Display name for the model.
            version_aliases: Aliases for the model version.
        """
        TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
        try:
            # Register Model in Vertex AI Model Registry
            model = aiplatform.Model.upload(
                display_name=display_name,
                model_id=f"model.keras",
                artifact_uri=self.config.root_dir,
                # serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest",
                is_default_version=True,
                version_aliases=version_aliases,
            )
            logger.info(f"Model registered successfully: {model.display_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            raise

    def evaluate(self):
        """
        Executes the evaluation process including data preparation, model evaluation, and deployment.
        """
        aiplatform.start_run(
            run=uuid.uuid4().hex,
        )
        try:
            X_test, y_test, y_test_labels = self.prep_data_for_evaluation()
            y_pred = self.model.predict(X_test)
            y_pred_one_hot = to_categorical(y_pred.argmax(axis=1), num_classes=7)
            y_true_label = np.argmax(y_test, axis=1)
            y_pred_label = np.argmax(y_pred_one_hot, axis=1)
            logger.info(f'Predictions: {y_pred_one_hot.shape}, Actuals: {y_test.shape}')
            logger.info(f'Predictions: {y_pred_one_hot[0]}, Actuals: {y_test[0]}')
            logger.info(f'Test Labels: {y_test_labels}')
            metrics, conf_matrix = self.evaluate_model(y_test, y_pred_one_hot, y_true_label, y_pred_label)
            print("Evaluation Metrics:", metrics)
            plot = self.plot_confusion_matrix(conf_matrix, y_test_labels)

            aiplatform.log_metrics(metrics)
            aiplatform.log_params({"model_version": "v1", "model_type": "classification"})

            plt_path = "confusion_matrix.png"
            plot.savefig(plt_path)
            plot.close()
            # aiplatform.log_artifact(plt_path)

            model = self.register_model("SER_CNN", ["v1"])

        except Exception as e:
            logger.error(f"Error during model prediction or evaluation: {str(e)}")
            raise

        finally:
            aiplatform.end_run()

    def authenticate(self):
        credentials = service_account.Credentials.from_service_account_file(
            "gcp_key.json"
        )
        return credentials
