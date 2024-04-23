from pathlib import Path
from utils import read_yaml, create_directories
from config_entity import DataTransformationConfig


class ConfigurationManager:
    """
    Class for managing configuration.

    Summary:
        This class handles the management of configuration files and provides methods to retrieve specific configuration objects.

    Explanation:
        The ConfigurationManager class reads and manages configuration files for data ingestion, validation, transformation, model training, and model evaluation.
        The class takes optional file paths for the configuration, parameters, and schema files.
        It provides methods to retrieve specific configuration objects for data ingestion, data validation, data transformation, model training, and model evaluation.

    Methods:
        get_data_ingestion_config() -> List[DataIngestionConfig]:
            Retrieves a list of DataIngestionConfig objects for each data ingestion configuration.

        get_data_validation_config() -> DataValidationConfig:
            Retrieves the DataValidationConfig object for data validation.

        get_data_transformation_config() -> DataTransformationConfig:
            Retrieves the DataTransformationConfig object for data transformation.

        get_model_trainer_config() -> ModelTrainerConfig:
            Retrieves the ModelTrainerConfig object for model training.

        get_model_evaluation_config() -> ModelEvaluationConfig:
            Retrieves the ModelEvaluationConfig object for model evaluation.

    Raises:
        None.

    Examples:
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_validation_config = config_manager.get_data_validation_config()
        data_transformation_config = config_manager.get_data_transformation_config()
        model_trainer_config = config_manager.get_model_trainer_config()
        model_evaluation_config = config_manager.get_model_evaluation_config()
    """

    def __init__(
        self,
        config_filepath=Path('./config.yaml'),
        params_filepath=Path('./params.yaml'),
        # schema_filepath='',
    ):
        """
        Class for configuration management.

        Summary:
            This class handles the management of configuration parameters.

        Explanation:
            The ConfigurationManager class is responsible for reading and managing configuration parameters.
            It initializes the configuration, parameters, and schema from the specified file paths.
            It also creates the necessary directories for artifacts.

        Attributes:
            config (dict): The configuration parameters.
            params The file (dict): The parameter values.
            schema (dict): The schema definition.

        Methods:
            None.
        """

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        # self.schema = read_yaml(schema_filepath)

        # create_directories([self.config.artifacts_root])

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Method for retrieving the data transformation configuration.

        Summary:
            This method retrieves the data transformation configuration.

        Explanation:
            The get_data_transformation_config() method returns a DataTransformationConfig object representing the data transformation configuration.
            It creates a DataTransformationConfig object using the specified configuration parameters.

        Returns:
            DataTransformationConfig: The data transformation configuration.

        Raises:
            None.
        """

        config = self.config.data_transformation

        # create_directories([config.root_dir,])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            metadata_path=config.metadata_path,
            metadata_train_path=config.metadata_train_path,
            metadata_test_path=config.metadata_test_path,
            output_path=config.output_path,
            train_path=config.train_path,
            val_path=config.val_path,
            test_path=config.test_path,
        )

        return data_transformation_config
