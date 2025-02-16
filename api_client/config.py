import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API credentials
API_KEY = os.getenv("IDEALISTA_API_KEY")
CLIENT_SECRET = os.getenv("IDEALISTA_CLIENT_SECRET")

if not API_KEY or not CLIENT_SECRET:
    raise ValueError(
        "Missing required environment variables: 'IDEALISTA_API_KEY' or 'IDEALISTA_CLIENT_SECRET'"
    )

# API usage tracking file
API_USAGE_FILE = "api_usage.json"

# Token URL for Idealista API
TOKEN_URL = "https://api.idealista.com/oauth/token"

# Base URL for Idealista API
BASE_URL = "https://api.idealista.com/3.5/"

# Default search parameters
DEFAULT_SEARCH_PARAMS = {
    "country": "pt",
    "locale": "pt",
    "language": "pt",
    "maxItems": "50",
    "operation": "rent",
    "propertyType": "homes",
    "order": "publicationDate",
    "sort": "desc",
    "maxPrice": "100000000",
    "sinceDate": "W",
}

# Predefined centers (cities) with their coordinates
CITY_COORDINATES = {
    "lisbon": "38.736946,-9.142685",
    "madrid": "40.416775,-3.703790",
    "barcelona": "41.385064,-2.173404",
}

# Predefined idealista location IDs for cities
CITY_LOCATION_IDS = {
    "lisbon": "0-EU-PT-11-06",
    "madrid": "0-EU-ES-28-07-001-079",
    "barcelona": "0-EU-ES-08-13-001-019",
}


def get_search_params(
    city: str = "lisbon", distance_to_center: Optional[int] = None, **overrides
) -> dict:
    """
    Returns search parameters for the specified city with optional overrides.

    Args:
        city (str): The city to search (case-insensitive, e.g., 'lisbon', 'madrid', 'barcelona').
        distance_to_center (int): Optional. The distance in meters to the city center. If not provided, the city's location ID will be used.
        **overrides: Additional search parameters to override defaults.

    Returns:
        dict: Search parameters for the API call.
    """
    if city.lower() not in CITY_COORDINATES:
        raise ValueError(
            f"City '{city}' is not configured. Available cities: {list(CITY_COORDINATES.keys())}"
        )

    search_params = DEFAULT_SEARCH_PARAMS.copy()

    if distance_to_center is not None:
        if distance_to_center <= 0:
            raise ValueError("distance_to_center must be a positive integer.")
        search_params.update(
            {
                "center": CITY_COORDINATES[city.lower()],
                "distance": distance_to_center,
            }
        )
    else:
        search_params.update(
            {
                "locationId": CITY_LOCATION_IDS[city.lower()],
            }
        )

    search_params.update(overrides)  # Apply overrides if provided

    return search_params
