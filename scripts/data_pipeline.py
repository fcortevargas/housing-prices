"""
Data pipeline script for extracting, cleaning, and preprocessing housing data.

This script serves as the first stage in the ML pipeline, handling:
1. Data extraction from API
2. Initial cleaning and standardization
3. Outlier detection and handling
4. Saving cleaned data for the feature pipeline
"""

import argparse
import logging
import datetime
from typing import Dict, List, Optional, Any

import pandas as pd

from api_client.api import IdealistaAPIClient
from utils.data_handlers import IdealistaDataSaver, IdealistaDataLoader

# Set up logging
from config.logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


class DataPipeline:
    def __init__(
        self, config_file: Optional[str] = None, date_str: Optional[str] = None
    ):
        """
        Initialize the data pipeline with configuration settings.

        Args:
            config_file: Path to the API config file
            date_str: Optional date string for the file
        """

        # Create API client
        self.client = IdealistaAPIClient(config_file)

        if date_str is None:
            date_str = datetime.date.today().isoformat()

        self.raw_data_saver = IdealistaDataSaver(
            base_path=self.client.config.get_data_extraction_config("raw_data_path"),
            city=self.client.config.get_search_params("city"),
            operation=self.client.config.get_search_params("operation"),
            date_or_unioned=date_str,
        )

        self.raw_data_loader = IdealistaDataLoader(
            base_path=self.client.config.get_data_extraction_config("raw_data_path"),
            city=self.client.config.get_search_params("city"),
            operation=self.client.config.get_search_params("operation"),
            date_or_unioned=date_str,
        )

        self.cleaned_data_saver = IdealistaDataSaver(
            base_path=self.client.config.get_data_extraction_config("cleaned_data_path"),
            city=self.client.config.get_search_params("city"),
            operation=self.client.config.get_search_params("operation"),
            date_or_unioned=date_str,
        )

    def extract_data(self) -> None:
        """
        Extract data from the Idealista API.

        Args:
            config_file: Path to the API config file
        """
        logger.info("Starting data extraction process")
        
        # Extract data for each city
        results = self.client.fetch_data()

        if not results:
            logging.warning("No results to save")
            return

        # Save raw listings
        self.raw_data_saver.save_json_data(results)

        # Convert results to DataFrame
        results_df = self.raw_data_saver.results_to_df(results)

        # Save processed listings
        self.raw_data_saver.save_csv_data(results_df)

        logger.info("Data extraction completed successfully")

    def clean_data(self) -> None:
        """
        Clean the raw data and prepare it for feature engineering.

        Args:
            base_path: Base directory for data storage
            city: City name (e.g., "lisbon")
            operation: Operation type (e.g., "rent" or "sale")
            date_str: Date string to use for the file

        Returns:
            Path to the cleaned data file
        """
        logger.info(f"Starting data cleaning process")

        df = self.raw_data_loader.load_csv_data()

        logger.info(f"Loaded {len(df)} rows of raw data")

        df.drop_duplicates(inplace=True)

        self.cleaned_data_saver.save_csv_data(df)


def validate_date(date_str):
    """
    Validate the date string format.

    Args:
        date_str: Date string to validate

    Returns:
        str: Validated date string in YYYY-MM-DD format

    Raises:
        argparse.ArgumentTypeError: If the date string is not in the correct format
    """
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError("Date must be in YYYY-MM-DD format")


def main():
    """Main function to run the data pipeline."""
    parser = argparse.ArgumentParser(description="Idealista data pipeline")
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract data from the Idealista API",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean the data",
    )
    parser.add_argument(
        "--extract-clean",
        action="store_true",
        help="Extract and clean the data from the Idealista API",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to data extraction configuration file",
    )
    parser.add_argument(
        "--date",
        type=validate_date,
        help="Date string for the data files (default: today)",
    )

    args = parser.parse_args()

    data_pipeline = DataPipeline(config_file=args.config, date_str=args.date)

    # Step 1: Extract data (if needed)
    if args.extract:
        data_pipeline.extract_data()

    # Step 2: Clean the data
    if args.clean:
        data_pipeline.clean_data()

    # Combine Steps 1 & 2
    if args.extract_clean:
        data_pipeline.extract_data()
        data_pipeline.clean_data()

    logger.info("Data pipeline completed successfully")


if __name__ == "__main__":
    main()
