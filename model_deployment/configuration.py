from pathlib import Path
from utils import read_yaml, create_directories
from dotenv import load_dotenv
from config_entity import DeploymentConfig

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
        config_filepath=Path("./config.yaml")
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


    def get_model_deployment_config(self) -> DeploymentConfig:
        """
        Method for retrieving the model deployment configuration.

        Summary:
            This method retrieves the model deployment configuration.

        Explanation:
            The get_model_deployment_config() method returns a DeploymentConfig object representing the model deployment configuration.
            It creates a DeploymentConfig object using the specified configuration parameters.

        Returns:
            DeploymentConfig: The model deployment configuration.

        Raises:
            None.
        """

        config = self.config.model_deploy

        model_deploy_config = DeploymentConfig(
            model_name=config.model_name,
            machine_type=config.machine_type,
            min_replica_count=config.min_replica_count,
            max_replica_count=config.max_replica_count,
            traffic_percentage=config.traffic_percentage,
            scaler_path=config.scaler_path
        )

        return model_deploy_config
