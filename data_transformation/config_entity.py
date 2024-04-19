from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Data class for data transformation configuration.

    Summary:
        This data class represents the configuration for data transformation.

    Explanation:
        The DataTransformationConfig class is a frozen data class that holds the configuration parameters for data transformation.
        It contains the root directory path, metadata path, output path, train path, and test path for data transformation.

    Attributes:
        root_dir (Path): The root directory path for data transformation.
        metadata_path (Path): The path to the metadata file.
        output_path (Path): The output path for the transformed data.
        train_path (Path): The path for the train data.
        test_path (Path): The path for the test data.

    Examples:
        config = DataTransformationConfig(root_dir, metadata_path, output_path, train_path, test_path)
    """

    root_dir: Path
    metadata_path: Path
    metadata_train_path: Path
    metadata_test_path: Path
    output_path: Path
    train_path: Path
    val_path: Path
    test_path: Path
