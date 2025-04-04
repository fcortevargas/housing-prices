"""
Configuration management for the Idealista API client.
"""

import os
from typing import Dict, Optional, Any
from pathlib import Path
import yaml
from dotenv import load_dotenv

base_dir = Path(__file__).resolve().parent.parent

# Default API config
DEFAULT_data_extraction_config = {
    "token_url": "https://api.idealista.com/oauth/token",
    "base_url": "https://api.idealista.com/3.5/",
    "max_retries": 3,
    "min_days_between_similar_requests": 7,
    "max_pages": None,
    "monthly_quota": 100,
    "usage_file": Path(__file__).resolve().parent / "api_usage.json",
    "raw_data_path": Path(__file__).resolve().parent.parent / "data" / "idealista" / "raw",
    "cleaned_data_path": Path(__file__).resolve().parent.parent / "data" / "idealista" / "cleaned",
}

# Default search parameters
DEFAULT_PARAMS = {
    "city": "lisbon",
    "order": "publicationDate",
    "sort": "desc",
    "maxPrice": "100000000",
    "operation": "rent",
    "propertyType": "homes",
    "sinceDate": "W",
    "country": "pt",
    "locale": "pt",
    "language": "pt",
    "maxItems": "50",
}

# City coordinates and location IDs
CITY_COORDINATES = {
    "lisbon": "38.736946,-9.142685",
    "madrid": "40.416775,-3.703790",
    "barcelona": "41.385064,-2.173404",
}

CITY_LOCATION_IDS = {
    "lisbon": "0-EU-PT-11-06",
    "madrid": "0-EU-ES-28-07-001-079",
    "barcelona": "0-EU-ES-08-13-001-019",
}


class IdealistaAPIConfig:
    def __init__(
        self,
        config_file: Optional[str] = None,
        env_file_path: Optional[str] = base_dir / ".env",
    ):
        self.config_file = config_file
        self.env_file_path = env_file_path
        self._load_config()

        if not self.__data_extraction_config or not self.__search_params:
            raise ValueError("Invalid configuration.")

    def _load_config(self) -> Dict:
        """
        Load configuration from YAML file and merge with defaults.

        Args:
            config_file: Optional path to config file

        Returns:
            Dict: Complete configuration
        """
        # Start with default params
        config = {
            "api": DEFAULT_data_extraction_config.copy(),
            "search": DEFAULT_PARAMS.copy(),
        }

        # Load overrides if provided
        if self.config_file:
            with open(self.config_file) as f:
                overrides = yaml.safe_load(f)
                config["api"].update(overrides.get("api", {}))
                config["search"].update(overrides.get("search", {}))

        # Validate API config
        self._validate_data_extraction_config(config["api"])

        # Load credentials from environment
        if not load_dotenv(self.env_file_path):
            raise ValueError("Failed to load environment variables.")

        # Validate credentials
        self._validate_credentials()

        # Parse city parameter to search params
        city = config["search"]["city"].lower()

        # Validate city
        if city not in CITY_COORDINATES:
            raise ValueError(
                f"City '{city}' is not configured. Available cities: {list(CITY_COORDINATES.keys())}"
            )

        # Set attributes
        self.__data_extraction_config = config["api"]
        self.__search_params = config["search"]

    def _validate_data_extraction_config(self, data_extraction_config: Dict[str, Any]):
        """
        Validates required parameters for the API configuration.

        Args:
            data_extraction_config: Dictionary of API configuration parameters

        Raises:
            ValueError: If required parameters are missing
        """
        required = [
            "token_url",
            "base_url",
            "max_retries",
            "min_days_between_similar_requests",
            "max_pages",
            "monthly_quota",
            "usage_file",
            "raw_data_path",
            "cleaned_data_path",
        ]
        missing = [key for key in required if key not in data_extraction_config]
        if missing:
            raise ValueError(
                f"Missing required API configuration parameters: {', '.join(missing)}"
            )

    def _validate_credentials(self):
        """
        Validates the API credentials.

        Raises:
            ValueError: If required credentials are not set in the environment variables
        """
        if not os.getenv("IDEALISTA_API_KEY") or not os.getenv(
            "IDEALISTA_CLIENT_SECRET"
        ):
            raise ValueError(
                "Missing required API credentials: 'IDEALISTA_API_KEY' or 'IDEALISTA_CLIENT_SECRET'. Please set them in your .env file."
            )

    def _validate_search_params(self, search_params: Dict[str, Any]):
        """
        Validates required parameters for the search API.

        Args:
            params: Dictionary of query parameters

        Raises:
            ValueError: If required parameters are missing
        """
        required = ["country", "operation", "propertyType", "sinceDate"]

        missing = [key for key in required if key not in search_params]
        if missing:
            raise ValueError(
                f"Missing required API search parameters: {', '.join(missing)}"
            )

        if not (
            all(k in search_params for k in ["center", "distance"])
            or "locationId" in search_params
        ):
            raise ValueError(
                "Either 'center + distance' or 'locationId' must be specified in the search parameters."
            )

    def get_data_extraction_config(self, key: Optional[str] = None) -> Dict[str, Any] | Any:
        """
        Get API configuration.

        Args:
            key: Optional key to get a specific configuration value

        Returns:
            Dict[str, Any]: API configuration
        """
        if key:
            return self.__data_extraction_config.get(key)
        return self.__data_extraction_config

    def get_search_params(self, key: Optional[str] = None) -> Dict[str, Any] | Any:
        """
        Get search parameters.

        Args:
            key: Optional key to get a specific search parameter value

        Returns:
            Dict[str, Any]: Search parameters
        """
        if key:
            return self.__search_params.get(key)
        return self.__search_params

    def prepare_search_params(self):
        """
        Prepare search parameters.

        Returns:
            Dict[str, Any]: Prepared search parameters
        """
        # Copy search params
        search_params = self.__search_params.copy()

        # Pop city from search params
        city = search_params.pop("city")

        # Validate distance to center
        distance_to_center = search_params.get("distance")
        if distance_to_center is not None:
            if distance_to_center < 0:
                raise ValueError("distance must be a positive integer")

            # Update search params with center
            search_params.update(
                {
                    "center": CITY_COORDINATES[city],
                }
            )
        else:
            # Update search params with location ID
            search_params.update(
                {
                    "locationId": CITY_LOCATION_IDS[city],
                }
            )

        # Validate search params
        self._validate_search_params(search_params)

        # Return prepared search params
        return search_params
