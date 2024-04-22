import numpy as np
import pandas as pd
from scipy import stats
from logger import logger


class DriftDetector:
    """
    A class for detecting data drift across multiple features.

    Methods:
        calculate_psi(expected, actual, bucket_type="bins", buckets=10, axis=0): Calculate the PSI (population stability index) across all variables.
        calculate_ks_statistic(train_data, test_data, features): Calculate the Kolmogorov-Smirnov statistic for each feature.
        detect_data_drift(train_data, test_data, features, bucket_type="bins", buckets=10): Detect data drift across multiple features using PSI.
    """

    def __init__(self) -> None:
        pass

    def calculate_psi(self, expected, actual, bucket_type="bins", buckets=10, axis=0):
        """
        Calculate the PSI (population stability index) / Jeffreys Divergence across all variables.

        Args:
            expected (pd.Series): Data from the original model (training set).
            actual (pd.Series): New data to compare against the original model.
            bucket_type (str): Type of strategy for creating buckets, can be 'bins' for equal width or 'quantiles' for equal frequency.
            buckets (int): Number of buckets to use for calculating PSI.
            axis (int): Axis along which the PSI is calculated, default is 0.

        Returns:
            float: The PSI value.
        """
        if bucket_type == "bins":
            # Define the bin edges using the full range of both datasets
            bins = np.linspace(
                start=min(expected.min(), actual.min()),
                stop=max(expected.max(), actual.max()),
                num=buckets + 1,
            )
        elif bucket_type == "quantiles":
            # Define the bin edges using quantiles
            bins = np.quantile(
                np.concatenate([expected, actual]), np.linspace(0, 1, num=buckets + 1)
            )
        else:
            raise ValueError("Invalid bucket_type: choose 'bins' or 'quantiles'")

        expected_counts = np.histogram(expected, bins=bins)[0]
        actual_counts = np.histogram(actual, bins=bins)[0]

        # Avoid division by zero
        expected_counts = np.where(expected_counts == 0, 0.0001, expected_counts)
        actual_counts = np.where(actual_counts == 0, 0.0001, actual_counts)

        # Calculate the PSI
        psi_values = (actual_counts - expected_counts) * np.log(
            actual_counts / expected_counts
        )
        psi = np.sum(psi_values)

        return psi

    def calculate_ks_statistic(self, train_data, test_data, features):
        """
        Calculate the Kolmogorov-Smirnov statistic for each feature to detect data drift.

        Args:
            train_data (pd.DataFrame): Training data, containing the features to analyze.
            test_data (pd.DataFrame): New data to compare against the training data.
            features (list): List of feature names to calculate the K-S statistic.

        Returns:
            dict: A dictionary with feature names as keys and K-S statistic values as values.
        """
        ks_results = {}
        for feature in features:
            # Calculate the K-S statistic and p-value for each feature
            statistic, p_value = stats.ks_2samp(train_data[feature], test_data[feature])
            ks_results[feature] = {"K-S Statistic": statistic, "p-value": p_value}
            logger.info(
                f"K-S Statistic for {feature}: {statistic:.4f}, p-value: {p_value:.4f}"
            )

        return ks_results

    def detect_data_drift(
        self, train_data, test_data, features, bucket_type="bins", buckets=10
    ):
        """
        Detect data drift across multiple features using PSI.

        Args:
            train_data (pd.DataFrame): Training data, containing the features to analyze.
            test_data (pd.DataFrame): New data to compare against the training data.
            features (list): List of feature names to calculate PSI.
            bucket_type (str): Method to use for bucketing values in the PSI calculation.
            buckets (int): Number of buckets to use for the PSI calculation.
        """
        psi_results = {}
        for feature in features:
            psi = self.calculate_psi(
                train_data[feature],
                test_data[feature],
                bucket_type=bucket_type,
                buckets=buckets,
            )
            psi_results[feature] = psi
            logger.info(f"PSI for {feature}: {psi:.4f}")

        return psi_results
