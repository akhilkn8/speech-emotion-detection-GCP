import os
import google.cloud.aiplatform as aiplatform
from logger import logger
from google.oauth2 import service_account
from dotenv import load_dotenv


load_dotenv()

class ModelServing:
    def __init__(self):
        self.credentials = self.authenticate()
    
    def authenticate(self):
        credentials = service_account.Credentials.from_service_account_file(
            "gcp_key.json"
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
        try:
            endpoints = aiplatform.Endpoint.list(project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                                                 location=os.getenv("AIPLATFORM_LOCATION"),
                                                #  filter=f"display_name=endpoint-id", 
                                                 credentials=self.credentials)
            endpoint = endpoints[0]  # Use existing endpoint
            logger.info(f'Found endpoint {endpoint.display_name}')
            return endpoint
        
        except Exception as e:
            logger.error(f'Endpoint not found or no endpoints available.\n{e}')

