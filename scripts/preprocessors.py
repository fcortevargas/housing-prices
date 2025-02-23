import datetime
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer

from feature_engine.outliers import OutlierTrimmer
from feature_engine.imputation import AddMissingIndicator
from feature_engine.wrappers import SklearnTransformerWrapper


class IdealistaDataLoader:
    """
    Class to load the data from a dataset created with the Idealista API.

    Args:
        read_path (str): Path to the dataset
        city (str): City of the dataset (e.g. Madrid, Barcelona or Lisbon)
        operation (str): Operation of the dataset (e.g. sale or rent)
        date_or_unioned (str): Date or unioned of the dataset (e.g. 2020-01-01 or unioned)
        include_geodata (bool): Include geodata in the dataset
    """

    def __init__(
        self,
        read_path: str,
        city: str = "lisbon",
        operation: str = "rent",
        date_or_unioned: str = "unioned",
        include_geodata: bool = True,
        index_col: str | int = "propertyCode",
    ):
        if not (
            date_or_unioned == "unioned" or self.is_valid_date_format(date_or_unioned)
        ):
            raise ValueError(
                "date_or_unioned must be 'unioned' or a valid date in the format 'YYYY-MM-DD'"
            )

        self.read_path = read_path
        self.city = city
        self.operation = operation
        self.date_or_unioned = date_or_unioned
        self.include_geodata = include_geodata
        self.index_col = index_col

        self.file_name = (
            f"{self.date_or_unioned}-{self.city}-listings-for-{self.operation}"
        )
        if include_geodata:
            self.file_name += "-with-geodata"

    @staticmethod
    def is_valid_date_format(date_string: str) -> bool:
        """
        Check if the date string is in the format 'YYYY-MM-DD'
        """
        try:
            datetime.datetime.strptime(date_string, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def load_data(self) -> pd.DataFrame:
        """
        Load the data from the specified path

        Returns:
            pd.DataFrame: Data from the dataset
        """
        file_path = (
            Path(self.read_path)
            / "cleaned"
            / self.operation
            / self.city
            / f"{self.file_name}.csv"
        )
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        try:
            data = pd.read_csv(file_path, index_col=self.index_col)
        except Exception as e:
            raise IOError(f"Error reading the file {file_path}: {e}")
        return data


class IdealistaDataCleaner(BaseEstimator, TransformerMixin):
    """
    Class to clean the Idealista dataset with the given method and threshold.

    Args:
        variables (list): List of variables to remove outliers
        method (str): Method to clean the data (e.g. iqr, z-score, etc.)
        threshold (float): Threshold for the data cleaning method
    """

    def __init__(
        self, variables: list, method: str, threshold: float | None = None, **kwargs
    ):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.method = method
        self.threshold = threshold

        if self.method in ["iqr", "gaussian", "mad", "quantiles"]:
            self.cleaner = OutlierTrimmer(
                capping_method=self.method,
                fold=self.threshold,
                variables=self.variables,
                **kwargs,
            )
        elif self.method in ["frequent_value_amputing"]:
            self.cleaner = FrequentValueAmputer(
                variables=self.variables,
                threshold=self.threshold,
            )
        elif self.method in ["rare_value_trimming"]:
            self.cleaner = RareValueTrimmer(
                variables=self.variables,
                threshold=self.threshold,
            )
        elif self.method in ["missing_indicator"]:
            self.cleaner = AddMissingIndicator(
                variables=self.variables,
                **kwargs,
            )
        elif self.method in ["knn_imputation"]:
            self.cleaner = SklearnTransformerWrapper(
                transformer=KNNImputer(**kwargs),
                variables=self.variables,
            )
        else:
            raise NotImplementedError(f"Method '{self.method}' is not implemented.")

    def _concat_data(self, X: pd.DataFrame | None, y: pd.DataFrame | None = None):
        if X is not None and y is not None:
            data = pd.concat([X, y], axis=1).copy()
        elif X is None and y is not None:
            data = y.copy()
        else:
            data = X.copy()
        return data

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None):
        data = self._concat_data(X, y)
        self.cleaner.fit(data)
        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame | None = None):
        data = self._concat_data(X, y)
        data = self.cleaner.transform(data)
        return data


class FrequentValueAmputer(BaseEstimator, TransformerMixin):
    """
    Class to replace observations of a list of features from a dataset with overly frequent values with NaN.

    Args:
        variables (list): List of variables to remove frequent values from
        threshold (float): The frequency for the frequent value replace
    """

    def __init__(self, variables: list, threshold: float = 0.05):
        self.variables = variables
        self.threshold = threshold
        self.frequent_values = {}

    def fit(self, data: pd.DataFrame):
        for variable in self.variables:
            value_counts = data[variable].value_counts(normalize=True)
            self.frequent_values[variable] = value_counts[
                value_counts > self.threshold
            ].index.tolist()
        return self

    def transform(self, data: pd.DataFrame):
        data = data.copy()
        for variable in self.variables:
            frequent_indexes = data[variable].isin(self.frequent_values[variable])
            data.loc[frequent_indexes, variable] = np.nan
        return data


class RareValueTrimmer(BaseEstimator, TransformerMixin):
    """
    Class to remove rare values from a dataset based on a given frequency threshold.

    Args:
        variables (list): List of variables to remove rare values from
        threshold (float): Threshold for the rare value removal
    """

    def __init__(self, variables: list, threshold: float = 0.05):
        self.variables = variables
        self.threshold = threshold
        self.rare_values = {}

    def fit(self, data: pd.DataFrame):
        for variable in self.variables:
            value_counts = data[variable].value_counts(normalize=True)
            self.rare_values[variable] = value_counts[
                value_counts < self.threshold
            ].index.tolist()
        return self

    def transform(self, data: pd.DataFrame):
        data = data.copy()
        for variable in self.variables:
            rare_indexes = data[variable].isin(self.rare_values[variable])
            data = data[~rare_indexes]
        return data
