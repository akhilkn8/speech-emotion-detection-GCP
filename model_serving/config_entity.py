from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelServingConfig:
    """
    Data class for model trainer configuration.

    Summary:
        This data class represents the configuration for model training.

    Explanation:
        The ModelTrainerConfig class is a frozen data class that holds the configuration parameters for model training.
        It contains the root directory path, paths for train and test data, model name, model parameters, and target column name.

    Attributes:
        root_dir (Path): The root directory path for model training.
        train_data_path (Path): The path to the train data.
        test_data_path (Path): The path to the test data.
        model_name (str): The name of the model.
        model_params (dict): The parameters for the model.
        target_col (str): The name of the target column.

    Examples:
        config = ModelTrainerConfig(root_dir, train_data_path, test_data_path, model_name, model_params, target_col)
    """

    scaler_path: str
    encoder_path: str
