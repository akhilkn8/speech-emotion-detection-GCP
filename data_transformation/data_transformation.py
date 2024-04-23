import os
import warnings
import yaml
import timeit
import multiprocessing as mp
from multiprocessing import get_context
from itertools import repeat
from joblib import Parallel, delayed, dump, load

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logger import logger
from config_entity import DataTransformationConfig
from drift_detection import DriftDetector

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

    def __init__(self, frame_length=2048, hop_length=512):
        self.frame_length = frame_length
        self.hop_length = hop_length

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
        return result

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


class DataTransformation:
    """
    Class for data transformation.

    Summary:
        This class handles the transformation of audio data by applying various augmentation techniques and extracting features.

    Explanation:
        The DataTransformation class provides methods to transform audio data by adding noise, time-stretching, shifting, and changing pitch.
        It also includes methods for feature extraction from the transformed audio data.
        The class takes a DataTransformationConfig object as input, which contains the necessary configuration parameters for data transformation.
        The get_features() method loads audio data from a specified path and extracts features using the FeatureExtractor class.
        The process_feature() method processes a single audio file by extracting features and associating them with a specified emotion label.
        The feature_engineering() method performs data transformation and feature extraction on a dataset in parallel using multiprocessing.
        The train_test_split_data() method splits the transformed dataset into train and test sets.
        Transforms the data by augmenting it with standard audio augmentation transforms

        Justification:
        https://aws.amazon.com/what-is/data-augmentation/
        Audio transformations typically include injecting random or Gaussian noise into some audio,
        fast-forwarding parts, changing the speed of parts by a fixed rate, or altering the pitch.

    Args:
        config (DataTransformationConfig): The configuration object containing the necessary parameters for data transformation.

    Methods:
        get_features(path: str, duration: float = 2.5, offset: float = 0.6) -> np.ndarray:
            Loads audio data from the specified path, extracts features, and returns the feature array.

        process_feature(path: str, emotion: str) -> Tuple[List[np.ndarray], List[str]]:
            Processes a single audio file by extracting features and associating them with the specified emotion label.
            Returns the feature array and emotion labels.

        feature_engineering():
            Performs data transformation and feature extraction on the dataset in parallel using multiprocessing.

        train_test_split_data(test_size: float = 0.2):
            Splits the transformed dataset into train and test sets and saves them to disk.

    Raises:
        No transformation parameters specified: If no transformation parameters are specified in the configuration.

    Examples:
        data_transformation = DataTransformation(config)
        features = data_transformation.get_features(path)
        X, Y = data_transformation.process_feature(path, emotion)
        data_transformation.feature_engineering()
        data_transformation.train_test_split_data(test_size)
    """

    def __init__(self, config: DataTransformationConfig, stage="train"):
        """
        Class for data transformation.

        Summary:
            This class handles the transformation of data using audio augmentation and feature extraction techniques.

        Explanation:
            The DataTransformation class takes a DataTransformationConfig object as input and provides methods for audio augmentation and feature extraction.
            It initializes the AudioAugmenter and FeatureExtractor classes and uses the specified configuration parameters for data transformation.

        Args:
            config (DataTransformationConfig): The configuration object containing the necessary parameters for data transformation.

        Methods:
            None.

        Raises:
            None.
        """

        self.config = config
        self.stage = stage
        self.chunksize = 1000  # for processing data in chunks

        # Audio Augmentation & Feature Extraction
        self.aug = AudioAugmenter()
        self.feat = FeatureExtractor()
        self.drift_detector = DriftDetector()
        # read data augmentation params from config file
        # option to try multiple augmentation params and observe the influence on model performance
        with open("./params.yaml", "r") as f:
            tfx_params = yaml.safe_load(f)
        self.tfx_params = tfx_params["data_transforms"][f"{stage}_params"]

    def get_features(self, path, duration=2.5, offset=0.6):
        """
        Function for extracting features from audio data.

        Summary:
            This function extracts features from audio data using various transformation techniques.

        Explanation:
            The get_features() function takes an audio file path as input and extracts features from the audio data.
            It applies different transformation techniques such as adding noise, time stretching, and pitch shifting to the audio data.
            The function returns a feature array containing the extracted features.

        Args:
            path (str): The path to the audio file.
            duration (float): The duration of the audio segment to consider (default: 2.5 seconds).
            offset (float): The offset from the beginning of the audio file to start the segment (default: 0.6 seconds).

        Returns:
            np.ndarray: The feature array containing the extracted features.

        Raises:
            None.
        """

        # Load raw audio data
        data, sr = librosa.load(path, duration=duration, offset=offset)
        audio_feats = []
        # perform data augmentation
        for param in self.tfx_params:
            if param == "default":
                audio_feats.append(self.feat.extract_features(data))
            elif param == "noise":
                audio_feats.append(self.feat.extract_features(self.aug.noise(data)))
            elif param == "pitch":
                audio_feats.append(self.feat.extract_features(self.aug.pitch(data, sr)))
            elif param == "pitch_noise":
                pitch_audio = self.aug.pitch(data, sr)
                audio_feats.append(
                    self.feat.extract_features(self.aug.noise(pitch_audio))
                )
            elif param == "pitch_shift_noise":
                pitch_audio = self.aug.pitch(data, sr)
                shift_audio = self.aug.shift(pitch_audio)
                audio_feats.append(
                    self.feat.extract_features(self.aug.noise(shift_audio))
                )
            elif param == "pitch_shift_stretch_noise":
                pitch_audio = self.aug.pitch(data, sr)
                shift_audio = self.aug.shift(pitch_audio)
                stretch_audio = self.aug.stretch(shift_audio)
                audio_feats.append(
                    self.feat.extract_features(self.aug.noise(stretch_audio))
                )
            elif param == "shift_noise":
                shift_audio = self.aug.shift(data)
                audio_feats.append(
                    self.feat.extract_features(self.aug.noise(shift_audio))
                )
            elif param == "stretch":
                audio_feats.append(self.feat.extract_features(self.aug.stretch(data)))
            elif param == "stretch_noise":
                stretch_audio = self.aug.stretch(data)
                audio_feats.append(
                    self.feat.extract_features(self.aug.noise(stretch_audio))
                )
            elif param == "stretch_pitch_noise":
                pitch_audio = self.aug.pitch(data, sr)
                stretch_audio = self.aug.stretch(pitch_audio)
                audio_feats.append(
                    self.feat.extract_features(self.aug.noise(stretch_audio))
                )
            elif param == "stretch_shift_noise":
                stretch_audio = self.aug.stretch(data)
                shift_audio = self.aug.shift(stretch_audio)
                audio_feats.append(
                    self.feat.extract_features(self.aug.noise(shift_audio))
                )
            else:
                logger.error("No transformation parameters specified!")
        # stack and return augmented audio representing real world scenario
        audio = np.vstack(audio_feats)
        return audio

    def process_feature(self, path, emotion):
        """
        Function for processing a single audio feature.

        Summary:
            This function processes a single audio file by extracting features and associating them with a specified emotion label.

        Explanation:
            The process_feature() function takes an audio file path and an emotion label as input.
            It calls the get_features() function to extract features from the audio file and associates them with the specified emotion label.
            The function returns the feature array and emotion labels.

        Args:
            path (str): The path to the audio file.
            emotion (str): The emotion label associated with the audio file.

        Returns:
            Tuple[List[np.ndarray], List[str]]: The feature array and emotion labels.

        Raises:
            None.
        """

        features = self.get_features(path)
        X = []
        Y = []
        for ele in features:
            X.append(ele)
            Y.append(emotion)
        return X, Y

    def feature_engineering(self):  # sourcery skip: extract-duplicate-method
        """
        Function for feature engineering.

        Summary:
            This function performs feature engineering on audio data.

        Explanation:
            The feature_engineering() function takes the root directory and metadata path as input.
            It reads the metadata file, drops any rows with missing values, and extracts features from the audio data.
            The function saves the feature array to disk and returns descriptive statistics of the data.

        Args:
            root_dir (str): The root directory path.
            metadata_dir (str): The path to the metadata file.

        Returns:
            None.

        Raises:
            None.
        """

        if self.stage == 'train':
            metadata_dir = os.path.join(self.config.metadata_train_path)
        elif self.stage == 'test':
            metadata_dir = os.path.join(self.config.metadata_test_path)

        data = pd.read_csv(metadata_dir)
        data = data.dropna()
        paths = data["FilePath"]
        emotions = data["Emotions"]
        logger.warning(" ==== Using Multiprocessors for Data Transformation ====")
        logger.info(f"{mp.cpu_count()} CPUs available")
        start = timeit.default_timer()
        logger.info("Multiprocessing started!")
        logger.info(f"{len(paths)}")
        # Perform feature extraction in parallel using multiprocessing
        with get_context("spawn").Pool() as pool:
            results = pool.starmap(self.process_feature, zip(paths, emotions))

        logger.info(f"Pre-preprocessing done for {len(results)} files")
        elapsed_time = timeit.default_timer() - start
        logger.info(f"Elapsed Time: {elapsed_time:.2f} secs")
        # retrieve top 20 logs in accordance to total time spent descending

        logger.info("Trying to export dataset to disk....")
        start = timeit.default_timer()
        # Start processing data in chunks of self.chunksize to reduce IO overheads
        pqwriter = None
        outfile = os.path.join(
            f"{self.config.output_path}", f"{self.stage}_data.parquet"
        )
        for i, chunk_start in enumerate(range(0, len(results), self.chunksize)):
            logger.info(f"Processing Chunk {i} now...")
            chunk_end = min(chunk_start + self.chunksize, len(results))
            results_chunk = results[chunk_start:chunk_end]

            X_chunk = []
            Y_chunk = []
            # Unravel features (ndarrays) before creating dataframe
            for result in results_chunk:
                x, y = result
                X_chunk.extend(x)
                Y_chunk.extend(y)

            emotions_df = pd.DataFrame(X_chunk)
            emotions_df.fillna(
                0, inplace=True
            )  # fill missing value w/ 0 - short duration audio
            emotions_df["Emotions"] = Y_chunk  # Extract labels

            # Converted unstructured data -> structured data & storing as parquet for later use
            # emotions_df.to_parquet(
            #     f"{self.config.output_path}/data_part_{i}.parquet", compression="gzip"
            # )
            table = pa.Table.from_pandas(emotions_df)
            if i == 0:
                # create a parquet write object giving it an output file
                pqwriter = pq.ParquetWriter(outfile, table.schema)
            pqwriter.write_table(table)

        # close the parquet writer
        if pqwriter:
            pqwriter.close()

        logger.info("Dataframe written to disk!!")
        elapsed_time = timeit.default_timer() - start
        logger.info(f"Elapsed Time: {elapsed_time:.2f} secs")

    def get_data_paths(self, output_path):
        """
        Checks if the output directory exists and returns a list of data file paths matching the pattern.

        Args:
            output_path (str): The path to the directory containing data files.

        Returns:
            list: A list of data file paths.

        Raises:
            FileNotFoundError: If the output directory does not exist or no data files are found.
        """
        if not os.path.isdir(output_path):
            raise FileNotFoundError(f"Output directory {output_path} does not exist!")

        data_files = os.listdir(output_path)
        if data_paths := [
            os.path.join(output_path, filename)
            for filename in data_files
            if filename.startswith("data_part_") and filename.endswith(".parquet")
        ]:
            return data_paths

        else:
            raise FileNotFoundError("No data files found matching the pattern!")

    def train_test_split_data(self, test_size=0.2, val_size=0.2):
        """
        Splits the data into train, validation, and test sets.

        Args:
            test_size (float, optional): The proportion of the data to use for the test set. Defaults to 0.2.
            val_size (float, optional): The proportion of the data to use for the validation set. Defaults to 0.5.

        Returns:
            tuple: A tuple containing the train, validation, and test dataframes.
        """
        data_paths = self.get_data_paths(self.config.output_path)
        all_data = [pd.read_parquet(path) for path in data_paths]
        emotions_df = pd.concat(all_data, ignore_index=True)

        # Split into train and test sets
        train, test = train_test_split(
            emotions_df,
            test_size=test_size,
            stratify=emotions_df["Emotions"],
            random_state=42,
        )
        # Split train into train and validation
        train, val = train_test_split(
            train, test_size=val_size, stratify=train["Emotions"], random_state=42
        )
        logger.info(
            f"Shapes ==> Train: {train.shape}, Val: {val.shape}, Test: {test.shape}"
        )
        return train, val, test

    def split_and_scale(self, test_size, val_size, method="standard"):
        """
        Splits the data into train, validation, and test sets, and scales the features.

        Args:
            method (str, optional): The method to use for splitting and scaling. Defaults to "standard".
            test_size (float, optional): The proportion of the data to use for the test set. Defaults to 0.2.
            val_size (float, optional): The proportion of the data to use for the validation set. Defaults to 0.2.

        Raises:
            None.

        Examples:
            # Example 1: Split and scale the data using Z-Score normalization
            split_and_scale(test_size=0.2, val_size=0.2, method="standard")

            # Example 2: Split and scale the data using min-max normalization
            split_and_scale(test_size=0.2, val_size=0.2, method="min-max")
        """

        train_data, val_data, test_data = self.train_test_split_data(
            test_size=test_size, val_size=val_size
        )
        X_train = train_data.drop("Emotions", axis=1)
        X_val = val_data.drop("Emotions", axis=1)
        X_test = test_data.drop("Emotions", axis=1)
        y_train = train_data["Emotions"]
        y_val = val_data["Emotions"]
        y_test = test_data["Emotions"]
        if method == "standard":
            self.X_mean = np.mean(X_train, axis=0)
            self.X_std = np.std(X_train, axis=0)
            # Perform Z-Score Normalization (StandardScaler) by Training mean & std
            X_train = (X_train - self.X_mean) / self.X_std
            X_val = (X_val - self.X_mean) / self.X_std
            X_test = (X_test - self.X_mean) / self.X_std
        elif method == "min-max":
            self.X_min = np.min(X_train, axis=0)
            self.X_max = np.max(X_train, axis=0)
            # Perform Min-Max Normalization by Training min & max
            X_train = (X_train - self.X_min) / (self.X_max - self.X_min)
            X_val = (X_val - self.X_min) / (self.X_max - self.X_min)
            X_test = (X_test - self.X_min) / (self.X_max - self.X_min)
        else:
            logger.error("Unsupported or no scaling method provided!")
        X_train["Emotions"] = y_train
        X_val["Emotions"] = y_val
        X_test["Emotions"] = y_test
        X_train.to_parquet(self.config.train_path, compression="gzip")
        logger.info("Train data written to disk!!")
        X_val.to_parquet(self.config.val_path, compression="gzip")
        logger.info("Validation data written to disk!!")
        X_test.to_parquet(self.config.test_path, compression="gzip")
        logger.info("Test data written to disk!!")

    def scale_data(self, method="standard"):
        parquet_file = os.path.join(
            f"{self.config.output_path}", f"{self.stage}_data.parquet"
        )
        df = pd.read_parquet(parquet_file, engine="pyarrow")

        logger.info(
            f"Loaded {self.stage} file of size {df.shape} for scaling: {self.stage}_data.parquet"
        )
        mean_csv = os.path.join(f"{self.config.output_path}", "means.csv")
        std_csv = os.path.join(f"{self.config.output_path}", "stds.csv")

        if self.stage == "train":
            if method == "standard":
                # Find a store of means and std values
                # Create a new csv if not found
                try:
                    mean_df = pd.read_csv(mean_csv)
                except FileNotFoundError:
                    logger.info("No means.csv found. Creating ...")
                    mean_df = pd.DataFrame()
                try:
                    std_df = pd.read_csv(std_csv)
                except FileNotFoundError:
                    logger.info("No stds.csv found. Creating ...")
                    std_df = pd.DataFrame()

                scaler = StandardScaler()
                train_values = scaler.fit_transform(df.loc[:, df.columns != "Emotions"])
                train_values = np.hstack(
                    (train_values, df["Emotions"].values.reshape(-1, 1))
                )
                df_train = pd.DataFrame(train_values, columns=df.columns)
                scaler_outfile = os.path.join(
                    f"{self.config.output_path}", "std_scaler.bin"
                )
                dump(scaler, scaler_outfile, compress=True)
                mean_df = pd.concat(
                    [
                        mean_df,
                        pd.DataFrame(
                            scaler.mean_.reshape(1, -1),
                            columns=df.loc[:, df.columns != "Emotions"].columns,
                        ),
                    ],
                    axis=0,
                )
                # pd.concat([pd.DataFrame(), pd.DataFrame(scaler.mean_.reshape(1,-1), columns=x.columns)], axis=0)
                std_df = pd.concat(
                    [
                        std_df,
                        pd.DataFrame(
                            np.sqrt(scaler.var_.reshape(1, -1)),
                            columns=df.loc[:, df.columns != "Emotions"].columns,
                        ),
                    ]
                )
                mean_df.to_csv(mean_csv, index=False)
                std_df.to_csv(std_csv, index=False)
                df_train.to_parquet(parquet_file)

        elif self.stage == "test":
            if method == "standard":
                scaler_infile = os.path.join(
                    f"{self.config.output_path}", "std_scaler.bin"
                )
                scaler = load(scaler_infile)
                test_values = scaler.transform(df.loc[:, df.columns != "Emotions"])
                test_values = np.hstack(
                    (test_values, df["Emotions"].values.reshape(-1, 1))
                )
                df_test = pd.DataFrame(test_values, columns=df.columns)

            self.split_validation(df_test)

    def split_validation(self, test_df):
        logger.info(f"Splitting the {self.stage} file to test and validation")
        test_df, val_df = train_test_split(
            test_df, test_size=0.5, stratify=test_df["Emotions"], random_state=42
        )

        test_file = os.path.join(
            f"{self.config.output_path}", f"{self.stage}_data.parquet"
        )
        val_file = os.path.join(f"{self.config.output_path}", "val_data.parquet")

        test_df.to_parquet(test_file)
        logger.info(f"Created {self.stage}_data.parquet file")
        val_df.to_parquet(val_file)
        logger.info("Created val_data.parquet file")

    def evaluate_model_performance(self):

        train_data = pd.read_parquet(self.config.train_path, engine="pyarrow")
        test_data = pd.read_parquet(self.config.test_path, engine="pyarrow")
        features = test_data.columns
        # Detect drift using both K-S test and PSI
        ks_results = self.drift_detector.calculate_ks_statistic(
            train_data, test_data, features
        )
        psi_results = self.drift_detector.detect_data_drift(
            train_data, test_data, features
        )

        logger.info("K-S Test Results:", ks_results)
        logger.info("PSI Results:", psi_results)

        # Further actions based on drift results
        # E.g., retrain model, alert stakeholders, etc.
