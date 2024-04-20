from pathlib import Path
from utils import read_yaml, create_directories
import os
from dotenv import load_dotenv
from config_entity import ServingConfig

load_dotenv()


class ConfigurationManager:
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

    def __init__(
        self,
        config_filepath=Path("./config.yaml"),
        params_filepath=Path("./params.yaml"),
        schema_filepath=Path("./schema.yaml"),
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
        self.schema = read_yaml(schema_filepath)

        # create_directories([self.config.artifacts_root])

    def get_model_serving_config(self) -> ServingConfig:
        """
        Method for retrieving the model serving configuration.

        Summary:
            This method retrieves the model serving configuration.

        Explanation:
            The get_model_serving_config() method returns a ServingConfig object representing the model serving configuration.
            It creates a ServingConfig object using the specified configuration parameters.

        Returns:
            ServingConfig: The model serving configuration.

        Raises:
            None.
        """

        config = self.config.model_serve
        params = self.params.model_params
        label = self.schema.TARGET_COLUMN

        model_serve_config = ServingConfig(
            model_name=config.model_name,
            project_id=config.project_id,
            location=config.location,
            endpoint_id=config.endpoint_id,
            machine_type=config.machine_type,
            min_replica_count=config.min_replica_count,
            max_replica_count=config.max_replica_count,
            traffic_percentage=config.traffic_percentage,
        )

        return model_serve_config
