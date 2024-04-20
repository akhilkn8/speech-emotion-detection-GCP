import google.cloud.aiplatform as aiplatform
from config_entity import ServingConfig
from logger import logger


class ModelServing:
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
        aiplatform.init(project=self.config.project_id, location=self.config.location)

    def get_model_endpoint(self):
        """
        Retrieves or creates a model endpoint for serving.

        Args:
            endpoint_name: Name of the endpoint (default is "my_model_endpoint").

        Returns:
            Endpoint: The model endpoint for serving.
        """
        if endpoints := aiplatform.Endpoint.list(
            filter=f"display_name={self.config.endpoint_id}"
        ):
            endpoint = endpoints[0]  # Use existing endpoint
            print(f"Using existing endpoint: {endpoint.resource_name}")
        else:
            # Create a new endpoint if not exist
            endpoint = aiplatform.Endpoint.create(
                display_name=self.config.endpoint_id,
                labels={"env": "prod"},
            )
            print(f"Created new endpoint: {endpoint.resource_name}")
        return endpoint

    def get_model(self):
        """
        Retrieves the model from the Model Registry.

        Returns:
            Model: The machine learning model.
        """
        try:
            model = aiplatform.Model(model_name=self.config.model_name)
            return model
        except Exception as e:
            logger.error(
                "Could not find the model in Model Registry, please register the model first!"
            )

    def deploy_model_to_endpoint(self, endpoint):
        """
        Deploys the model to the specified endpoint for serving.

        Args:
            endpoint: The model endpoint to deploy the model.

        Returns:
            DeployedModel: The deployed model for serving.
        """
        model = self.get_model()
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
