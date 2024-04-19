from pathlib import Path
from utils import read_yaml, create_directories
import os
from dotenv import load_dotenv
from config_entity import ModelTrainerConfig

load_dotenv()

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
        schema_filepath=Path('./schema.yaml'),
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

        RUNTIME = os.environ.get('RUNTIME', 'GCP')
        if RUNTIME == 'local':
            config_filepath = Path(f'./config_local.yaml')

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        # create_directories([self.config.artifacts_root])


    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Method for retrieving the model trainer configuration.

        Summary:
            This method retrieves the model trainer configuration.

        Explanation:
            The get_model_trainer_config() method returns a ModelTrainerConfig object representing the model trainer configuration.
            It creates a ModelTrainerConfig object using the specified configuration parameters.

        Returns:
            ModelTrainerConfig: The model trainer configuration.

        Raises:
            None.
        """

        config = self.config.model_trainer
        params = self.params.model_params
        label = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_path=config.train_path,
            val_path=config.val_path,
            model_name=config.model_name,
            params=params,
            target_col=label,
        )

        return model_trainer_config

    