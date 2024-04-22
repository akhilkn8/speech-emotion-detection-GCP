from deployment import ModelDeployment
from configuration import ConfigurationManager
from logger import logger

STAGE_NAME = "model deployment stage"


class ModelDeploymentPipeline:
    """
    A pipeline for deployment a machine learning model for audio emotion recognition.

    Methods:
        main: Main function to run the audio emotion recognition pipeline.
    """

    def __init__(self):
        """
        Initializes the ModelDeploymentPipeline by setting up the model server for deployment.
        """

        config_manager = ConfigurationManager()
        model_server_config = config_manager.get_model_deployment_config()
        self.model_server = ModelDeployment(config=model_server_config)

    def main(self):
        """
        Main function to run the speech emotion recognition pipeline.

        Returns:
            None.
        """
        # Load model and make predictions
        endpoint = self.model_server.get_model_endpoint()
        deployed_model = self.model_server.deploy_model_to_endpoint(endpoint=endpoint)
        logger.info(f'Success! Deployed Endpoint and Model')


if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = ModelDeploymentPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e
