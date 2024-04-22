import numpy as np
import streamlit as st
from joblib import load
from configuration import ConfigurationManager
from data_transformation import FeatureExtractor
from serving import ModelServing
import librosa
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
        self.config = config_manager.get_model_serving_config()

    def main(self):
        """
        Main function to run the speech emotion recognition pipeline.

        Returns:
            None.
        """
        st.title("Audio Emotion Recognition")
        audio_file = st.file_uploader("Upload an audio file", type=["wav"])

        if audio_file is not None:
            data, sr = librosa.load(audio_file, duration=2.5, offset=0.6)

            # Process audio file
            feature_extractor = FeatureExtractor(config=self.config)
            features = feature_extractor.extract_features(data, sr)

            # Load model and make predictions
            endpoint = ModelServing().get_model_endpoint()
            prediction = endpoint.predict([np.repeat(features, 64, axis=0).T.tolist()])
            
            encoder = load(self.config.encoder_path)
            label = encoder.inverse_transform(prediction.predictions)

            st.write(f"Predicted Emotion:", label.item())


if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = ModelServingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e
