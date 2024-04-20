import streamlit as st
from serving import ModelServing
from data_transformation import FeatureExtractor, DataTransformation
from configuration import ConfigurationManager
from logger import logger

STAGE_NAME = "model serving stage"


class ModelServingPipeline:
    """
    A pipeline for serving a machine learning model for audio emotion recognition.

    Methods:
        main: Main function to run the audio emotion recognition pipeline.
    """

    def __init__(self):
        """
        Initializes the ModelServingPipeline by setting up the model server for serving.
        """

        config_manager = ConfigurationManager()
        model_server_config = config_manager.get_model_serving_config()
        self.model_server = ModelServing(config=model_server_config)

    def main(self):
        """
        Main function to run the speech emotion recognition pipeline.

        Returns:
            None.
        """
        st.title("Audio Emotion Recognition")
        audio_file = st.file_uploader("Upload an audio file", type=["wav"])

        if audio_file is not None:
            # Process audio file
            feature_extractor = FeatureExtractor()
            features = feature_extractor.extract_features(audio_file)

            # Load model and make predictions
            endpoint = self.model_server.get_model_endpoint()
            prediction = endpoint.predict(instances=[features])
            st.write("Predicted Emotion:", prediction)


if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = ModelServingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e
