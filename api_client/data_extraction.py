"""
Data extraction script for Idealista API.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from api_client.api import IdealistaAPIClient

from config.logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

base_dir = Path(__file__).resolve().parent.parent
config_dir = base_dir / "config"
config_file = config_dir / "data_extraction_config.yaml"

def main(config_file: Optional[str] = None):
    """
    Main function to extract data for configured cities.

    Args:
        config_file: Optional path to config file
    """

    # Create API client
    client = IdealistaAPIClient(config_file)

    # Extract data for each city
    client.fetch_data()


if __name__ == "__main__":
    main(config_file)

