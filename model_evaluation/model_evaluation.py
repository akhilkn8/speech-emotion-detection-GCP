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
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import numpy as np
import joblib
from utils import save_json
from pathlib import Path
from pathlib import Path
from mlcore.utils.common import save_json
from mlcore.entity.config_entity import ModelEvaluationConfig
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluation:
    """
    Class for model evaluation.

    Summary:
        This class handles the evaluation of a trained model using the specified configuration.

    Explanation:
        The ModelEvaluation class provides methods to evaluate a trained model.
        The class takes a ModelEvaluationConfig object as input, which contains the necessary configuration parameters for model evaluation.
        The evaluate_model() method calculates evaluation metrics such as accuracy, precision, recall, and F1-score.
        The log_into_mlflow() method logs the evaluation metrics and model parameters into MLflow for tracking and visualization.

    Args:
        config (ModelEvaluationConfig): The configuration object containing the necessary parameters for model evaluation.

    Methods:
        evaluate_model(ytrue: np.ndarray, ypred: np.ndarray) -> Tuple[float, float, float, float]:
            Evaluates the model by calculating accuracy, precision, recall, and F1-score.
            Returns the evaluation metrics.

        log_into_mlflow():
            Logs the evaluation metrics and model parameters into MLflow for tracking and visualization.

    Returns:
        Tuple[float, float, float, List[float]]: The evaluation metrics (accuracy, precision, recall, F1-score).
    """

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate_model(self, ytrue, ypred, yproba=None):
        accuracy = accuracy_score(ytrue, ypred)
        precision = precision_score(ytrue, ypred, average="macro")
        recall = recall_score(ytrue, ypred, average="macro")
        f1 = f1_score(ytrue, ypred, average="macro")
        report = classification_report(ytrue, ypred, output_dict=True)
        mcc = matthews_corrcoef(ytrue, ypred)
        conf_matrix = confusion_matrix(ytrue, ypred)
        return accuracy, precision, recall, f1, report, mcc, conf_matrix

    def plot_confusion_matrix(self, conf_matrix, class_names):
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

    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_path)
        model = joblib.load(self.config.model_path)

        xtest = test_data.drop(self.config.target_col, axis=1).values
        ytest = test_data[self.config.target_col].values

        os.environ["MLFLOW_TRACKING_URI"] = self.config.mlflow_uri
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            ypred = model.predict(xtest)

            (accuracy, precision, recall, f1, report, mcc, conf_matrix) = (
                self.evaluate_model(ytest, ypred)
            )

            # Plot and save confusion matrix
            plt = self.plot_confusion_matrix(conf_matrix, class_names)
            image_path = os.path.join(self.config.output_dir, "confusion_matrix.png")
            plt.savefig(image_path)
            plt.show()

            scores = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metrics(scores)

            mlflow.sklearn.log_model(model, "model")
