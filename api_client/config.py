import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API credentials
API_KEY = os.getenv("IDEALISTA_API_KEY")
CLIENT_SECRET = os.getenv("IDEALISTA_CLIENT_SECRET")

if not API_KEY or not CLIENT_SECRET:
    raise ValueError("API_KEY or CLIENT_SECRET are not set in the environment.")

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
    "distance": "90000"
}

# Predefined centers (cities) with their coordinates
CITY_COORDINATES = {
    "lisbon": "38.736946,-9.142685",
    "madrid": "40.416775,-3.703790",
    "barcelona": "41.385064,-2.173404",
}


def get_search_params(city, **overrides):
    """
    Returns search parameters for the specified city with optional overrides.

    Args:
        city (str): The city to search (e.g., 'lisbon', 'madrid', 'barcelona').
        **overrides: Additional search parameters to override defaults.

    Returns:
        dict: Search parameters for the API call.
    """
    if city.lower() not in CITY_COORDINATES:
        raise ValueError(
            f"City '{city}' is not configured. Available cities: {list(CITY_COORDINATES.keys())}"
        )

    search_params = DEFAULT_SEARCH_PARAMS.copy()
    search_params.update(
        {
            "center": CITY_COORDINATES[city.lower()],
        }
    )
    search_params.update(overrides)  # Apply overrides if provided
    return search_params
