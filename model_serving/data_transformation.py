import warnings
from joblib import Parallel, delayed, dump, load
import numpy as np
import librosa
from logger import logger

warnings.filterwarnings("ignore")

np.random.seed(42)  # For reproducibility

##  ===> Sorcery by Deb Begins! <====
## To understand this code, a broad background of Audio Digital Signal Processing (DSP) is required.
## If you are unfamiliar with audio signal processing, please refer to the following website before proceeding:
## https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
## NOTE: This code will use the multiprocessing library (All your CPU cores but one).


class AudioAugmenter:
    """
    Class for audio data augmentation.

    Summary:
        This class provides methods to augment audio data by adding noise, time-stretching, shifting, and changing pitch.

    Explanation:
        The AudioAugmenter class contains methods to apply various augmentation techniques to audio data.
        The noise() method adds additive white Gaussian noise (AWGN) to the audio data, providing a more realistic simulation of background noise or environmental conditions.
        The stretch() method time-stretches the audio data by a specified rate using the time_stretch function.
        The shift() method shifts the audio data by a random amount within a certain range, simulating changes in timing or alignment.
        The pitch() method changes the pitch of the audio data by a specified number of steps using the pitch_shift function.
        The class takes an optional noise_std parameter, which controls the standard deviation of the Gaussian noise added in the noise() method.

    Methods:
        noise(data: np.ndarray) -> np.ndarray:
            Adds additive white Gaussian noise (AWGN) to the audio data and returns the augmented data.

        stretch(data: np.ndarray, rate: float = 0.8) -> np.ndarray:
            Time-stretches the audio data by the specified rate and returns the augmented data.

        shift(data: np.ndarray) -> np.ndarray:
            Shifts the audio data by a random amount within a certain range and returns the augmented data.

        pitch(data: np.ndarray, sampling_rate: int, n_steps: int = 3) -> np.ndarray:
            Changes the pitch of the audio data by the specified number of steps and returns the augmented data.

    Args:
        data (np.ndarray): The input audio data.
        rate (float, optional): The rate of time-stretching. Default is 0.8.
        sampling_rate (int): The sampling rate of the audio data.
        n_steps (int, optional): The number of steps to change the pitch. Default is 3.

    Returns:
        np.ndarray: The augmented audio data.

    Examples:
        augmenter = AudioAugmenter()
        augmented_data = augmenter.noise(data)
    """

    def __init__(self, noise_std=0.035):
        self.noise_std = noise_std

    # NOISE
    def noise(self, data):
        """This method adds additive white Gaussian noise (AWGN) to the audio data.
        Gaussian noise can provide a more realistic simulation of background noise or environmental conditions that may affect speech signals
        """
        # amplitude-dependent additive noise
        # noise_amp = 0.035 * np.random.uniform() * np.amax(data)
        # data = data + noise_amp * np.random.normal(size=data.shape[0])
        noise = np.random.normal(scale=self.noise_std, size=data.shape[0])
        return data + noise

    # STRETCH
    def stretch(self, data, rate=0.8):
        """This method time-stretches the audio data by a specified rate.
        It uses the time_stretch function to perform time stretching with the specified rate.
        """
        return librosa.effects.time_stretch(data, rate=0.8)

    # SHIFT
    def shift(self, data):
        """This method shifts the audio data by a random amount within a certain range.
        It generates a random shift range using a uniform distribution and then rolls (shifts) the audio data by the generated amount.
        """
        shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
        return np.roll(data, shift_range)

    # PITCH
    def pitch(self, data, sampling_rate, n_steps=3):
        """This method changes the pitch of the audio data by a specified number of steps.
        It uses the pitch_shift function to shift the pitch of the audio data by the specified number of steps.
        """
        return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=n_steps)


class FeatureExtractor:
    """
    Class for feature extraction.

    Summary:
        This class provides methods to extract various audio features from input data.

    Explanation:
        The FeatureExtractor class contains methods to extract features such as zero-crossing rate (ZCR), root mean square energy (RMSE),
        Mel-frequency cepstral coefficients (MFCC), chromagram, Mel Spectrogram from audio data. The extract_features() method combines these features into a single array
        and returns the result. The class takes optional parameters for frame length and hop length, which control the size and overlap of the analysis windows.

    Methods:
        extract_features(data: np.ndarray, sr: int = 22050) -> np.ndarray:
            Extracts audio features from the input data and returns the combined feature array.

    Private Methods:
        __zcr__(data: np.ndarray) -> np.ndarray:
            Calculates the zero-crossing rate (ZCR) feature from the input data and returns the result.

        __rmse__(data: np.ndarray) -> np.ndarray:
            Calculates the root mean square energy (RMSE) feature from the input data and returns the result.

        __chroma__(data: np.ndarray, sr: int) -> np.ndarray:
            Calculates chromagram or power spectrogram from the input data and returns the result.

        __melspec__(data: np.ndarray, sr: int) -> np.ndarray:
            Calculates Mel Spectrogram from the input data and returns the result.

        __mfcc__(data: np.ndarray, sr: int, flatten: bool = True) -> np.ndarray:
            Calculates the Mel-frequency cepstral coefficients (MFCC) feature from the input data and returns the result.
            The flatten parameter determines whether to flatten the MFCC array or not. Default is True.

    Args:
        data (np.ndarray): The input audio data.
        sr (int, optional): The sample rate of the audio data. Default is 22050.

    Returns:
        np.ndarray: The combined feature array extracted from the input data.

    Examples:
        feature_extractor = FeatureExtractor()
        features = feature_extractor.extract_features(data)
    """

    def __init__(self, config, frame_length=2048, hop_length=512):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.config = config

    def extract_features(self, data, sr=22050):
        """Extracts audio features from the input data and returns the combined feature array."""
        result = np.array([])
        # Extracting features: ZCR, RMSE, and MFCC
        result = np.hstack(
            (
                result,
                self.__zcr__(data),
                self.__rmse__(data),
                self.__chroma__(data, sr),
                self.__melspec__(data, sr),
                self.__mfcc__(data, sr),
            )
        )
        transformed_result = self.scale_inference(result.reshape(1,-1))
        return transformed_result

    def scale_inference(self, input_data):
        scaler_infile = self.config.scaler_path

        scaler = load(scaler_infile)
        transformed_data = scaler.transform(input_data)
        return transformed_data

    def __zcr__(self, data):
        """Calculates the zero-crossing rate (ZCR) feature from the input data and returns the result."""
        zcr = np.mean(
            librosa.feature.zero_crossing_rate(
                data, frame_length=self.frame_length, hop_length=self.hop_length
            ).T,
            axis=0,
        )
        return np.squeeze(zcr)

    def __rmse__(self, data):
        """Calculates the root mean square energy (RMSE) feature from the input data and returns the result."""
        rmse = np.mean(
            librosa.feature.rms(
                y=data, frame_length=self.frame_length, hop_length=self.hop_length
            ).T,
            axis=0,
        )
        return np.squeeze(rmse)

    def __chroma__(self, data, sr):
        """Calculates chromagram or power spectrogram from the frequency data and returns the result."""
        # convert to time-frequency domain by Discrete Fourier Transform
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(
            librosa.feature.chroma_stft(S=stft, sr=sr, hop_length=self.hop_length).T,
            axis=0,
        )
        return np.squeeze(chroma_stft)

    def __melspec__(self, data, sr):
        """Calculates Mel Spectrogram from the input data and returns the result."""
        spectro = np.mean(
            librosa.feature.melspectrogram(y=data, sr=sr, hop_length=self.hop_length).T,
            axis=0,
        )
        return np.squeeze(spectro)

    def __mfcc__(self, data, sr, flatten=True):
        """Calculates the Mel-frequency cepstral coefficients (MFCC) feature from the input data and returns the result.
        The flatten parameter determines whether to flatten the MFCC array or not. Default is True.
        """
        mfccs = np.mean(
            librosa.feature.mfcc(
                y=data,
                sr=sr,
                n_mfcc=128,
                n_fft=self.frame_length,
                hop_length=self.hop_length,
            ).T,
            axis=0,
        )
        return np.ravel(mfccs) if flatten else np.squeeze(mfccs)
