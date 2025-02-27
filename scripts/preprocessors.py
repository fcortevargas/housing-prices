import datetime
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

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

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """
        Fit the cleaning transformer on the provided dataset.

        Parameters:
            X (pd.DataFrame): Input features.
            y (pd.Series or None): (Ignored) Optional target variable.

        Returns:
            self: The fitted transformer.
        """
        self.cleaner.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None):
        """
        Apply the cleaning transformation to the dataset.

        Parameters:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: The cleaned feature DataFrame.
        """
        if y is not None:
            return self.transform_x_y(X, y)
        else:
            return self.cleaner.transform(X)

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None):
        """
        Apply the cleaning transformation to both features and target (if provided).

        If the underlying cleaner provides a transform_x_y method, that is used; otherwise, y is aligned
        to the transformed X.

        Parameters:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.

        Returns:
            tuple: (X_transformed, y_transformed) where y_transformed is aligned with X_transformed.
        """
        if hasattr(self.cleaner, "transform_x_y"):
            return self.cleaner.transform_x_y(X, y)
        else:
            X_transformed = self.cleaner.transform(X)
            y_transformed = y.loc[X_transformed.index]
            return X_transformed, y_transformed


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
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.threshold = threshold
        self.frequent_values = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """
        Identify the overly frequent values for each specified variable.

        Parameters:
            X (pd.DataFrame): The dataset on which to compute frequency counts.
            y (pd.Series or None): (Ignored) Optional target variable.

        Returns:
            self: The fitted transformer.
        """
        for variable in self.variables:
            value_counts = X[variable].value_counts(normalize=True)
            self.frequent_values[variable] = value_counts[
                value_counts > self.threshold
            ].index.tolist()
        return self

    def transform(self, X: pd.DataFrame):
        """
        Replace overly frequent values with NaN in the dataset.

        Parameters:
            X (pd.DataFrame): The dataset to transform.

        Returns:
            pd.DataFrame: The transformed dataset with replaced values.
        """
        data = X.copy()
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
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.threshold = threshold
        self.rare_values = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """
        Identify the rare values for each specified feature.

        Parameters:
            X (pd.DataFrame): The dataset on which to compute frequency counts.
            y (pd.Series or None): (Ignored) Optional target variable.

        Returns:
            self: The fitted transformer.
        """
        for variable in self.variables:
            value_counts = X[variable].value_counts(normalize=True)
            self.rare_values[variable] = value_counts[
                value_counts < self.threshold
            ].index.tolist()
        return self

    def transform(self, X: pd.DataFrame):
        """
        Remove rows containing rare values from the dataset.

        Parameters:
            X (pd.DataFrame): The dataset to transform.

        Returns:
            pd.DataFrame: The dataset with rows containing rare values removed.
        """
        data = X.copy()
        for variable in self.variables:
            rare_indexes = data[variable].isin(self.rare_values[variable])
            data = data[~rare_indexes]
        return data

    def transform_x_y(self, X: pd.DataFrame, y: pd.Series):
        """
        Remove rows containing rare values from the dataset and align the target accordingly.

        Parameters:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.

        Returns:
            tuple: (X_transformed, y_transformed) where y_transformed is aligned with X_transformed.
        """
        X_transformed = self.transform(X)
        y_transformed = y.loc[X_transformed.index]
        return X_transformed, y_transformed


class UnaryFeatureDropper(BaseEstimator, TransformerMixin):
    """
    A transformer that drops features with only one unique value.

    Parameters:
        None
    """

    def __init__(self):
        self.features_to_drop = []

    def fit(self, X, y=None):
        """
        Identify features with only one unique value.

        Parameters:
            X (pd.DataFrame): The dataset from which to identify features.
            y (pd.Series or None): (Ignored) Optional target variable.

        Returns:
            self: The fitted transformer.
        """
        self.features_to_drop = X.columns[X.nunique() == 1].tolist()
        return self

    def transform(self, X):
        """
        Drop features with only one unique value from the dataset.

        Parameters:
            X (pd.DataFrame): The dataset from which to drop features.

        Returns:
            pd.DataFrame: The dataset with features dropped.
        """
        return X.drop(columns=self.features_to_drop)

    def transform_x_y(self, X, y):
        """
        Drop features with only one unique value from the dataset and align the target accordingly.

        Parameters:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.

        Returns:
            tuple: (X_transformed, y_transformed) where y_transformed is aligned with X_transformed.
        """
        X_transformed = self.transform(X)
        y_transformed = y.loc[X_transformed.index]
        return X_transformed, y_transformed


class RelativeFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    A transformer that generates relative features based on groupby columns.

    Parameters:
        variables (list): List of dictionaries with 'groupby' and 'target' keys.

        The 'groupby' key should contain the column name to group by, and the 'target' key
        should contain the column name to generate the relative feature for.
    """

    def __init__(self, variables: list):
        self._validate_variables(variables)
        self.variables = variables
        self.groupby_col = None
        self.target_col = None
        self.aggregate_func = None
        self.target_col = None

    def _validate_variables(self, variables: list):
        """
        Check if the variables list contains dictionaries with 'groupby', 'target', 'agg' and 'name' keys.

        Parameters:
            variables (list): List of variables to validate.

        Raises:
            ValueError: If any variable is not a dictionary with 'groupby', 'target', 'agg' and 'name' keys.
        """
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        required_keys = {"groupby", "target", "agg", "name"}
        for variable in variables:
            if not all(key in variable for key in required_keys):
                raise ValueError(
                    "Each variable should be a dictionary with 'groupby', 'target', 'agg' and 'name' keys"
                )

    def fit(self, X, y=None):
        """
        Empty fit method to comply with the scikit-learn API.

        Parameters:
            X (pd.DataFrame): The dataset from which to generate relative features.
            y (pd.Series or None): (Ignored) Optional target variable.

        Returns:
            self: The fitted transformer
        """
        return self

    def transform(self, X):
        """
        Generate relative features based on the specified groupby and target columns.

        Parameters:
            X (pd.DataFrame): The dataset from which to generate relative features.

        Returns:
            pd.DataFrame: The dataset with relative features added.
        """
        for variable in self.variables:
            self.groupby_col = variable["groupby"]
            self.target_col = variable["target"]
            self.aggregate_func = variable["agg"]
            self.feature_name = variable["name"]
            X = self._generate_relative_feature(X)
        return X

    def _generate_relative_feature(self, X):
        """
        Generate a relative feature based on the groupby and target columns.

        Parameters:
            X (pd.DataFrame): The dataset from which to generate the relative feature.

        Returns:
            pd.DataFrame: The dataset with the relative feature added.
        """
        X = X.copy()
        X[self.feature_name] = X.groupby(self.groupby_col)[self.target_col].transform(
            lambda x: x / x.agg(self.aggregate_func)
        )
        return X


class ProximityScoreGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, variables: list):
        self._validate_variables(variables)
        self.variables = variables
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _validate_variables(self, variables: list):
        """
        Check if the variables list contains dictionaries with 'target', 'name' keys.

        Parameters:
            variables (list): List of variables to validate.

        Raises:
            ValueError: If any variable is not a dictionary with target', 'name' keys.
        """
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        required_keys = {"target", "name"}
        for variable in variables:
            if not all(key in variable for key in required_keys):
                raise ValueError(
                    "Each variable should be a dictionary with 'target', 'name' keys."
                )

    def fit(self, X, y=None):
        """
        Empty fit method to comply with the scikit-learn API.

        Parameters:
            X (pd.DataFrame): The dataset from which to generate relative features.
            y (pd.Series or None): (Ignored) Optional target variable.

        Returns:
            self: The fitted transformer
        """
        return self

    def transform(self, X):
        """
        Generate relative features based on the specified groupby and target columns.

        Parameters:
            X (pd.DataFrame): The dataset from which to generate relative features.

        Returns:
            pd.DataFrame: The dataset with relative features added.
        """
        for variable in self.variables:
            self.target_col = variable["target"]
            self.feature_name = variable["name"]
            X = self._generate_proximity_score_feature(X)
        return X

    def _generate_proximity_score_feature(self, X):
        """
        Generate a relative feature based on the groupby and target columns.

        Parameters:
            X (pd.DataFrame): The dataset from which to generate the relative feature.

        Returns:
            pd.DataFrame: The dataset with the relative feature added.
        """
        X = X.copy()
        proximity = 1 - self.scaler.fit_transform(
            pd.DataFrame(X[self.target_col].min(axis=1))
        )
        X[self.feature_name] = proximity.mean(axis=1)
        return X
