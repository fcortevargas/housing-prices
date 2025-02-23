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
    A class for loading datasets created from the Idealista API.

    This class constructs a file path based on the provided parameters (city, operation,
    date, and geodata inclusion) and loads the CSV data into a pandas DataFrame.

    Parameters:
        read_path (str): Base directory where the dataset files are stored.
        city (str): City associated with the dataset (e.g., "Madrid", "Barcelona", "Lisbon").
        operation (str): Type of operation (e.g., "sale" or "rent").
        date_or_unioned (str): Either a date string in "YYYY-MM-DD" format or "unioned" to indicate combined data.
        include_geodata (bool): Flag indicating whether to include geodata in the file name.
        index_col (str or int): Column name or index to set as the DataFrame index.

    Raises:
        ValueError: If `date_or_unioned` is not "unioned" or a valid date string.
        FileNotFoundError: If the constructed file path does not exist.
        IOError: If an error occurs while reading the CSV file.
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
        Check if the provided date string is in the 'YYYY-MM-DD' format.

        Parameters:
            date_string (str): The date string to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            datetime.datetime.strptime(date_string, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def load_data(self) -> pd.DataFrame:
        """
        Load the CSV dataset from the constructed file path.

        The method builds the file path using the given parameters, verifies its existence,
        and reads the CSV into a pandas DataFrame using the specified index column.

        Returns:
            pd.DataFrame: The loaded dataset.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If an error occurs during file reading.
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
    A scikit-learn transformer for cleaning the Idealista dataset using various methods.

    This transformer supports multiple cleaning strategies including outlier trimming,
    frequent value amputation (replacing values with NaN), rare value trimming (dropping rows),
    adding missing value indicators, and KNN-based imputation.

    Parameters:
        variables (list): List of feature names on which to apply the cleaning process.
        method (str): The cleaning method to use. Options include:
                      - "iqr", "gaussian", "mad", "quantiles": Outlier trimming.
                      - "frequent_value_amputing": Replace overly frequent values with NaN.
                      - "rare_value_trimming": Remove rows containing rare values.
                      - "missing_indicator": Add a missing indicator column.
                      - "knn_imputation": Impute missing values using KNN.
        threshold (float, optional): Threshold used by the chosen cleaning method (e.g., IQR fold or frequency threshold).
        **kwargs: Additional keyword arguments passed to the underlying cleaning transformer.
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
        """
        Concatenate feature (X) and target (y) data if both are provided.

        Parameters:
            X (pd.DataFrame or None): Feature data.
            y (pd.DataFrame or None): Target data.

        Returns:
            pd.DataFrame: The concatenated DataFrame.
        """
        if X is not None and y is not None:
            data = pd.concat([X, y], axis=1).copy()
        elif X is None and y is not None:
            data = y.copy()
        else:
            data = X.copy()
        return data

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None):
        """
        Fit the cleaning transformer on the provided dataset.

        Parameters:
            X (pd.DataFrame): Input features.
            y (pd.DataFrame or None): Optional target variable.

        Returns:
            self: The fitted transformer.
        """
        data = self._concat_data(X, y)
        self.cleaner.fit(data)
        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame | None = None):
        """
        Apply the cleaning transformation to the dataset.

        Parameters:
            X (pd.DataFrame): Input features.
            y (pd.DataFrame or None): Optional target variable.

        Returns:
            pd.DataFrame: The cleaned dataset.
        """
        data = self._concat_data(X, y)
        data = self.cleaner.transform(data)
        return data


class FrequentValueAmputer(BaseEstimator, TransformerMixin):
    """
    A transformer that replaces overly frequent values in specified variables with NaN.

    This transformer identifies values in each specified variable that occur with a normalized
    frequency higher than the given threshold, and replaces those values with NaN to mark them
    for later imputation.

    Parameters:
        variables (list): List of variable names to process.
        threshold (float): Frequency threshold above which values will be replaced with NaN.
    """

    def __init__(self, variables: list, threshold: float = 0.05):
        self.variables = variables
        self.threshold = threshold
        self.frequent_values = {}

    def fit(self, data: pd.DataFrame):
        """
        Identify the overly frequent values for each specified variable.

        Parameters:
            data (pd.DataFrame): The dataset on which to compute frequency counts.

        Returns:
            self: The fitted transformer.
        """
        for variable in self.variables:
            value_counts = data[variable].value_counts(normalize=True)
            self.frequent_values[variable] = value_counts[
                value_counts > self.threshold
            ].index.tolist()
        return self

    def transform(self, data: pd.DataFrame):
        """
        Replace overly frequent values with NaN in the dataset.

        Parameters:
            data (pd.DataFrame): The dataset to transform.

        Returns:
            pd.DataFrame: The transformed dataset with replaced values.
        """
        data = data.copy()
        for variable in self.variables:
            frequent_indexes = data[variable].isin(self.frequent_values[variable])
            data.loc[frequent_indexes, variable] = np.nan
        return data


class RareValueTrimmer(BaseEstimator, TransformerMixin):
    """
    A transformer that removes rows containing rare values from specified features.

    This transformer identifies rare values in each specified feature based on a normalized
    frequency threshold and drops any rows that contain such rare values.

    Parameters:
        variables (list): List of feature names to process.
        threshold (float): Frequency threshold below which values are considered rare.
    """

    def __init__(self, variables: list, threshold: float = 0.05):
        self.variables = variables
        self.threshold = threshold
        self.rare_values = {}

    def fit(self, data: pd.DataFrame):
        """
        Identify the rare values for each specified feature.

        Parameters:
            data (pd.DataFrame): The dataset on which to compute frequency counts.

        Returns:
            self: The fitted transformer.
        """
        for variable in self.variables:
            value_counts = data[variable].value_counts(normalize=True)
            self.rare_values[variable] = value_counts[
                value_counts < self.threshold
            ].index.tolist()
        return self

    def transform(self, data: pd.DataFrame):
        """
        Remove rows containing rare values from the dataset.

        Parameters:
            data (pd.DataFrame): The dataset to transform.

        Returns:
            pd.DataFrame: The dataset with rows containing rare values removed.
        """
        data = data.copy()
        for variable in self.variables:
            rare_indexes = data[variable].isin(self.rare_values[variable])
            data = data[~rare_indexes]
        return data
