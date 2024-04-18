from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataGenerationConfig:
    """
    Data class for data generation configuration.

    Summary:
        This data class represents the configuration for data generation.

    Explanation:
        The DataIngestionConfig class is a frozen data class that holds the configuration parameters for data ingestion.
        It contains the root directory path, source URL, and local data path for data ingestion.

    Attributes:
        root_dir (Path): The root directory path for data ingestion.
        source_URL (str): The URL of the data source.
        local_data_path (Path): The local path where the data will be saved after ingestion.

    Examples:
        config = DataIngestionConfig(root_dir, source_URL, local_data_path)
    """

    gcp_metadata_bucket: str
    gcp_train_bucket: str
    gcp_test_bucket: str
    metadata_dir: Path
    train_dir: Path
    test_dir: Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Data class for data ingestion configuration.

    Summary:
        This data class represents the configuration for data ingestion.

    Explanation:
        The DataIngestionConfig class is a frozen data class that holds the configuration parameters for data ingestion.
        It contains the root directory path, source URL, and local data path for data ingestion.

    Attributes:
        root_dir (Path): The root directory path for data ingestion.
        source_URL (str): The URL of the data source.
        local_data_path (Path): The local path where the data will be saved after ingestion.

    Examples:
        config = DataIngestionConfig(root_dir, source_URL, local_data_path)
    """

    root_dir: Path
    source_URL: str
    local_data_path: Path
    gcp_bucket_name: str
    gcp_data_path: Path



@dataclass(frozen=True)
class ModelTrainerConfig:
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

    root_dir: Path
    train_path: Path
    val_path: Path
    model_name: str
    params: dict
    target_col: str
