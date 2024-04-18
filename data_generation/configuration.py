from pathlib import Path
import utils as utils
from config_entity import DataGenerationConfig


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
        # params_filepath=PARAMS_FILE_PATH,
        # schema_filepath=SCHEMA_FILE_PATH,
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

        self.config = utils.read_yaml(config_filepath)
        # self.params = utils.read_yaml(params_filepath)
        # self.schema = utils.read_yaml(schema_filepath)

        # utils.create_directories([self.config.artifacts_root])
    
    def get_data_generation_config(self) -> DataGenerationConfig:
        
        config = self.config.data_generation
        utils.create_directories([config.metadata_dir])
        utils.create_directories([config.train_dir])
        utils.create_directories([config.test_dir])
        
        data_generation_config = DataGenerationConfig(
            gcp_metadata_bucket = config.gcp_metadata_bucket,
            gcp_train_bucket = config.gcp_train_bucket,
            gcp_test_bucket = config.gcp_test_bucket,
            metadata_dir = config.metadata_dir,
            train_dir = config.train_dir,
            test_dir = config.test_dir,
            )
        return data_generation_config
