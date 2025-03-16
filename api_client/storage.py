from pathlib import Path
import json
import logging
import datetime
import pandas as pd
from typing import List, Dict, Any


class DataStorage:
    """
    Handles safe storage of API data with backup mechanisms.

    This class ensures that data is saved safely by:
    1. Using temporary files for atomic writes
    2. Separating raw and processed data
    3. Providing clear error handling
    """

    def __init__(self, base_path: str = "data"):
        """
        Initialize the storage system.

        Args:
            base_path (str): Base directory for all data storage
        """
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.interim_path = self.base_path / "interim"
        self.processed_path = self.base_path / "processed"
        self._ensure_directories()

    def _ensure_directories(self):
        """Create all necessary directories if they don't exist."""
        for path in [self.raw_path, self.interim_path, self.processed_path]:
            path.mkdir(parents=True, exist_ok=True)

    def save_raw_listings(
        self, listings: List[Dict[str, Any]], city: str, operation: str
    ) -> bool:
        """
        Safely save raw listings data in JSON format.

        Args:
            listings: List of dictionaries containing the raw listing data
            city: Name of the city
            operation: Type of operation (e.g., 'sale', 'rent')

        Returns:
            bool: True if save was successful, False otherwise
        """
        if not listings:
            logging.warning("No listings to save")
            return False

        date_str = datetime.date.today().isoformat()
        filename = f"{date_str}-{city}-listings-for-{operation}.json"
        temp_file = self.raw_path / f"temp_{filename}"
        final_file = self.raw_path / filename

        try:
            # First save to temporary file
            with open(temp_file, "w") as f:
                json.dump(listings, f, indent=2)

            # If successful, move to final location
            temp_file.rename(final_file)
            logging.info(f"Successfully saved raw data to {final_file}")
            return True

        except Exception as e:
            logging.error(f"Failed to save listings: {e}")
            if temp_file.exists():
                temp_file.unlink()  # Clean up temp file if it exists
            return False

    def save_processed_listings(
        self, df: pd.DataFrame, city: str, operation: str
    ) -> bool:
        """
        Safely save processed listings to CSV.

        Args:
            df: DataFrame containing the processed listing data
            city: Name of the city
            operation: Type of operation (e.g., 'sale', 'rent')

        Returns:
            bool: True if save was successful, False otherwise
        """
        if df.empty:
            logging.warning("Empty DataFrame - nothing to save")
            return False

        date_str = datetime.date.today().isoformat()
        filename = f"{date_str}-{city}-listings-for-{operation}.csv"
        temp_file = self.processed_path / f"temp_{filename}"
        final_file = self.processed_path / filename

        try:
            # Save to temporary file first
            df.to_csv(temp_file, index=False)
            # If successful, move to final location
            temp_file.rename(final_file)
            logging.info(f"Successfully saved processed data to {final_file}")
            return True

        except Exception as e:
            logging.error(f"Failed to save processed listings: {e}")
            if temp_file.exists():
                temp_file.unlink()  # Clean up temp file if it exists
            return False
