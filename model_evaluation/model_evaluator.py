import os
from datetime import datetime
import uuid
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
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
import tensorflow as tf
import keras
from dotenv import load_dotenv


load_dotenv()


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

    def __init__(
        self,
        config: ModelEvaluationConfig,
        experiment_name: str = "speech-emotion-evaluation",
    ):
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
            # self.model = load_model(self.config.model_path)  # load keras model
            self.model = tf.saved_model.load(self.config.model_path)  # load keras model
            # self.model = keras.layers.TFSMLayer(
            #     self.config.model_path, call_endpoint="serving_default"
            # )  # load keras model
            logger.info(
                f"Model has been successfully loaded from {self.config.model_path}"
            )
        except Exception as e:
            logger.error(f"Error loading Model: {str(e)}")
            raise
        self.credentials = self.authenticate()
        try:
            aiplatform.init(
                experiment=os.getenv("AIPLATFORM_EXPERIMENT"),
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("AIPLATFORM_LOCATION"),
                staging_bucket=os.getenv("AIPLATFORM_BUCKET"),
                credentials=self.credentials,
            )
            logger.info("AI Platform initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize AI Platform: {str(e)}")
            raise
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
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

    def plot_confusion_matrix(self, conf_matrix, class_names=None):
        """
        Plots a confusion matrix for model evaluation.

        Args:
            conf_matrix: Confusion matrix data.
            class_names: Names of classes for the confusion matrix.
        """
        plot = plt.figure(figsize=(20, 20))
        # sns.heatmap(
        #     conf_matrix,
        #     annot=True,
        #     fmt="g",
        #     cmap="Blues",
        #     xticklabels=self.encoder.categories_,
        #     yticklabels=self.encoder.categories_,
        # )
        cmd = ConfusionMatrixDisplay(conf_matrix, display_labels=self.encoder.categories_[0])
        plot = cmd.plot()
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        return plot

    def register_model(self, display_name, version_aliases):
        """
        Registers the model to a model registry.

        Args:
            display_name: Display name for the model.
            version_aliases: Aliases for the model version.
        """
        
        try:
            # Register Model in Vertex AI Model Registry
            model = aiplatform.Model.upload(
                display_name=display_name,
                model_id=f"model_{display_name}_{self.timestamp}",
                artifact_uri=self.config.model_path,
                serving_container_image_uri=os.getenv('SERVING_CONTAINER_URI'),
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
            y_pred = self.model.serve(X_test).numpy()
            logger.info(f'ypred: {y_pred}')
            y_pred_one_hot = to_categorical(y_pred.argmax(axis=1), num_classes=7)
            y_true_label = np.argmax(y_test, axis=1)
            y_pred_label = self.encoder.inverse_transform(y_pred_one_hot)
            # y_pred_label = np.argmax(y_pred_one_hot, axis=1)
            logger.info(f"Predictions: {y_pred_one_hot.shape}, Actuals: {y_test.shape}")
            logger.info(f"Predictions: {y_pred_one_hot[0]}, Actuals: {y_test[0]}")
            logger.info(f"y_test_labels: {y_test_labels}")
        
            y_test_labels = [x for x in y_test_labels]
            y_pred_label = [x for x in y_pred_label]
            logger.info(f"y_true_label: {y_true_label}")
            logger.info(f"y_pred_label: {y_pred_label}")
            
            metrics, conf_matrix = self.evaluate_model(
                y_test, y_pred_one_hot, y_test_labels, y_pred_label
            )

            print("Evaluation Metrics:", metrics)
            plot = self.plot_confusion_matrix(conf_matrix)

            aiplatform.log_metrics(metrics)
            aiplatform.log_params(
                {"model_version": "v1", "model_type": "classification"}
            )

            plt_path = os.path.join(self.config.confusion_matrix_path, f"confusion_matrix_{self.timestamp}.png")
            plot.figure_.savefig(plt_path)
            # plot.close()
            logger.info(f'Saved Confusion Matrix: {plt_path}')
            # aiplatform.log_artifact(plt_path)

            model = self.register_model("cnn-latest", ["v1"])

        except Exception as e:
            logger.error(f"Error during model prediction or evaluation: {str(e)}")
            raise

        finally:
            aiplatform.end_run()

    def authenticate(self):
        credentials = service_account.Credentials.from_service_account_file(os.getenv('GCP_CREDENTIAL_PATH'))
        return credentials
