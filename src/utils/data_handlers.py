from pathlib import Path
import json
import logging
import datetime
import pandas as pd
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


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


class IdealistaDataLoader:
    """
    A class for loading datasets created from the Idealista API.

    This class constructs a file path based on the provided parameters (city, operation,
    date, and geodata inclusion) and loads the CSV data into a pandas DataFrame.

    Parameters:
        base_path (str): Base directory where the dataset files are stored.
        city (str): City associated with the dataset (e.g., "Madrid", "Barcelona", "Lisbon").
        operation (str): Type of operation (e.g., "sale" or "rent").
        date (str): A date string in "YYYY-MM-DD" format to indicate which dataset will be loaded.
        include_geodata_features (bool): Flag to select the data file with geodata features in it.
        include_llm_features (bool): Flag to select the data file with LLM features in it.
        index_col (str or int): Column name or index to set as the DataFrame index.

    Raises:
        ValueError: If `date` is not "unioned" or a valid date string.
        FileNotFoundError: If the constructed file path does not exist.
        IOError: If an error occurs while reading the CSV file.
    """

    def __init__(
        self,
        base_path: str,
        city: str = "lisbon",
        operation: str = "rent",
        date: str | None = None,
        include_geodata_features: bool = False,
        include_llm_features: bool = False,
        index_col: str | int = "propertyCode",
    ):
        if date:
            self.set_file_name() # Set the file name based on the provided parameters
            if not is_valid_date_format(date):
                raise ValueError(
                    "`date` must be a valid date in the format 'YYYY-MM-DD'"
                )

        self.base_path = Path(base_path)
        self.city = city
        self.operation = operation
        self.date = date
        self.include_geodata_features = include_geodata_features
        self.include_llm_features = include_llm_features
        self.index_col = index_col

        self.load_path = self.base_path / self.city / self.operation

    def set_file_name(self) -> None:
        """
        Set the file name based on the provided parameters.

        The file name is constructed using the date or unioned flag, city, operation,
        and flags for geodata and LLM features.

        Returns:
            None: The method modifies the instance variable `file_name` directly.

        Raises:
            ValueError: If `date` is None.
        """
        if not self.date:
            raise ValueError("date is None")

        self.file_name = f"{self.date}-{self.city}-listings-for-{self.operation}"

        if self.include_geodata_features or self.include_llm_features:
            self.file_name += "-with"
        if self.include_geodata_features:
            self.file_name += "-geodata"
        if self.include_llm_features:
            self.file_name += "-llm"

    def load_json_data(self) -> pd.DataFrame:
        """
        Load the dataset from the constructed JSON file path.

        The method builds the file path using the given parameters, verifies its existence,
        and reads the file into a pandas DataFrame using the specified index column.

        Returns:
            pd.DataFrame: The loaded dataset.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If an error occurs during file reading.
        """
        file = self.file_name + ".json"
        file_path = self.load_path / file

        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            if self.index_col in df:
                df.set_index(self.index_col, inplace=True)
        except Exception as e:
            raise IOError(f"Error reading the file {self.load_path}: {e}")

        return df

    def load_csv_data(self) -> pd.DataFrame:
        """
        Load the dataset from the constructed CSV file path.

        The method builds the file path using the given parameters, verifies its existence,
        and reads the file into a pandas DataFrame using the specified index column.

        Returns:
            pd.DataFrame: The loaded dataset.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If an error occurs during file reading.
        """
        file = self.file_name + ".csv"
        file_path = self.load_path / file

        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        try:
            df = pd.read_csv(file_path)
            if self.index_col in df.columns:
                df.set_index(self.index_col, inplace=True)
        except Exception as e:
            raise IOError(f"Error reading the file {self.load_path}: {e}")

        return df


class IdealistaDataSaver:
    """
    A class for saving datasets created from the Idealista API.

    This class constructs a file path based on the provided parameters (city, operation,
    date, and geodata inclusion) and saves the pandas DataFrame into a CSV file in the given data stage.

    Parameters:
        base_path (str): Base directory where the dataset files will be saved.
        city (str): City associated with the dataset (e.g., "Madrid", "Barcelona", "Lisbon").
        operation (str): Type of operation (e.g., "sale" or "rent").
        date (str): Either a date string in "YYYY-MM-DD" format or "unioned" to indicate combined data.
        include_geodata_features (bool): Flag to indicate that the DataFrame has geodata features in it.
        include_llm_features (bool): Flag to indicate that the DataFrame has LLM features in it.
        index_col (str or int): Column name or index to set as the DataFrame index.

    Raises:
        ValueError: If `date` is not "unioned" or a valid date string.
        FileNotFoundError: If the constructed file path does not exist.
        IOError: If an error occurs while reading the CSV file.
    """

    def __init__(
        self,
        base_path: str,
        city: str = "lisbon",
        operation: str = "rent",
        date: str = datetime.date.today().isoformat(),
        include_geodata_features: bool = False,
        include_llm_features: bool = False,
        index_col: str | int = "propertyCode",
    ):
        if date:
            self.set_file_name() # Set the file name based on the provided parameters
            if not is_valid_date_format(date):
                raise ValueError(
                    "`date` must be a valid date in the format 'YYYY-MM-DD'"
                )

        self.base_path = Path(base_path)
        self.city = city
        self.operation = operation
        self.date = date
        self.include_geodata_features = include_geodata_features
        self.include_llm_features = include_llm_features
        self.index_col = index_col

        self.save_path = self.base_path / self.city / self.operation
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def set_file_name(self) -> None:
        """
        Set the file name based on the provided parameters.

        The file name is constructed using the date or unioned flag, city, operation,
        and flags for geodata and LLM features.

        Returns:
            None: The method modifies the instance variable `file_name` directly.

        Raises:
            ValueError: If `date` is None.
        """
        if not self.date:
            raise ValueError("date is None")

        self.file_name = f"{self.date}-{self.city}-listings-for-{self.operation}"

        if self.include_geodata_features or self.include_llm_features:
            self.file_name += "-with"
        if self.include_geodata_features:
            self.file_name += "-geodata"
        if self.include_llm_features:
            self.file_name += "-llm"

    def save_json_data(self, data: List[Dict[str, Any]]) -> bool:
        """
        Safely save listings data in JSON format.

        Args:
            data: List of dictionaries containing the raw listing data

        Returns:
            bool: True if save was successful, False otherwise
        """
        file = self.file_name + ".json"
        temp_file = self.save_path / f"temp_{file}"
        final_file = self.save_path / file

        logging.info(f"Saving data to {final_file}")

        if not data:
            logging.warning("No data to save")
            return False

        if not (
            isinstance(data, list) and all(isinstance(item, dict) for item in data)
        ):
            logging.warning("Input data is not a list of dictionaries")
            return False

        try:
            # First save to temporary file
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)

            # If successful, move to final location
            temp_file.rename(final_file)
            logging.info(f"Successfully saved data to {final_file}")
            return True

        except Exception as e:
            logging.error(f"Failed to save listings: {e}")
            if temp_file.exists():
                temp_file.unlink()  # Clean up temp file if it exists
            return False

    def results_to_df(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert API results to a pandas DataFrame.

        Args:
            results: List of property listings

        Returns:
            pd.DataFrame: DataFrame containing the listings
        """
        # Convert JSON to DataFrame
        df = pd.DataFrame.from_records(results)

        # Set the index of the DataFrame
        if self.index_col in df.columns:
            df.set_index(self.index_col, inplace=True)

        return df

    def save_csv_data(self, data: pd.DataFrame) -> bool:
        """
        Safely save listings data to CSV.

        Args:
            data: DataFrame containing the processed listing data

        Returns:
            bool: True if save was successful, False otherwise
        """
        file = self.file_name + ".csv"
        temp_file = self.save_path / f"temp_{file}"
        final_file = self.save_path / file

        logging.info(f"Saving data to {final_file}")

        if not isinstance(data, pd.DataFrame):
            logging.warning("Input data is not a dataframe")
            return False

        if data.empty:
            logging.warning("Empty DataFrame - nothing to save")
            return False

        try:
            # Check if index is set according to the configuration
            if self.index_col != data.index.name:
                logging.warning(
                    f"Configured index column: {self.index_col} is not the index of the DataFrame"
                )
                index = False
            else:
                index = self.index_col

            # Save the DataFrame to the temporary file
            data.to_csv(temp_file, index=index)

            # If successful, move to final location
            temp_file.rename(final_file)

            logging.info(f"Successfully saved processed data to {final_file}")

            return True

        except Exception as e:

            logging.error(f"Failed to save processed listings: {e}")
            if temp_file.exists():
                temp_file.unlink()  # Clean up temp file if it exists

            return False
