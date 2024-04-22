import os
import google.cloud.aiplatform as aiplatform
from google.oauth2 import service_account
from logger import logger
from dotenv import load_dotenv


load_dotenv()

class ModelDeployment:
    """
    A class for serving a machine learning model.

    Methods:
        get_model_endpoint: Retrieves or creates a model endpoint for serving.
        get_model: Retrieves the model from the Model Registry.
        deploy_model_to_endpoint: Deploys the model to the specified endpoint for serving.
    """

    def __init__(self, config):
        """
        Initializes the ModelServing instance with the provided configuration.
        """
        self.config = config
        aiplatform.init(project=os.getenv('GOOGLE_CLOUD_PROJECT'), location=os.getenv('AIPLATFORM_LOCATION'))
        self.credentials = self.authenticate()

    def authenticate(self):
        credentials = service_account.Credentials.from_service_account_file(
            os.getenv('GCP_CREDENTIAL_PATH')
        )
        return credentials

    def get_model_endpoint(self):
        """
        Retrieves or creates a model endpoint for serving.

        Args:
            endpoint_name: Name of the endpoint (default is "my_model_endpoint").

        Returns:
            Endpoint: The model endpoint for serving.
        """
        if endpoints := aiplatform.Endpoint.list(
            filter=f"display_name={os.getenv('ENDPOINT_NAME')}"
        ):
            endpoint = endpoints[0]  # Use existing endpoint
            logger.info(f"Using existing endpoint: {endpoint.resource_name}")
        else:
            # Create a new endpoint if not exist
            endpoint = aiplatform.Endpoint.create(
                display_name=os.getenv('ENDPOINT_NAME'),
                labels={"env": "prod"},
            )
            logger.info(f"Created new endpoint: {endpoint.resource_name}")
        logger.info(
            f"Model successfully deployed to endpoint: {endpoint.resource_name}"
        )
        return endpoint

    def get_model(self):
        """
        Retrieves the model from the Model Registry.

        Returns:
            Model: The machine learning model.
        """
        try:
            model_list = aiplatform.Model.list(project=os.getenv("GOOGLE_CLOUD_PROJECT"), 
                                               location=os.getenv("AIPLATFORM_LOCATION"), 
                                               credentials=self.credentials, 
                                               order_by="create_time desc")
            model = model_list[0]
            logger.info(f"Model retrieved: {model.display_name}")
            return model
        except Exception as e:
            logger.error(
                "Could not find the model in Model Registry, please register the model first!"
            )
            raise ValueError("Model not found in the Model Registry.") from e

    def deploy_model_to_endpoint(self, endpoint):
        """
        Deploys the model to the specified endpoint for serving.

        Args:
            endpoint: The model endpoint to deploy the model.

        Returns:
            DeployedModel: The deployed model for serving.
        """
        model = self.get_model()
        if not model:
            raise ValueError("Model could not be retrieved from the Model Registry.")
        deployed_model = endpoint.deploy(
            model=model,
            deployed_model_display_name=model.display_name,
            machine_type=self.config.machine_type,
            min_replica_count=self.config.min_replica_count,
            max_replica_count=self.config.max_replica_count,
            traffic_percentage=self.config.traffic_percentage,
            sync=True,
        )
        return deployed_model
