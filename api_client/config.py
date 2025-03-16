"""
Configuration management for the Idealista API client.

This module handles loading and validating configuration from environment
variables and provides structured access to API settings and parameters.
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class APIConfig:
    """API configuration settings."""
    api_key: str
    client_secret: str
    base_url: str = "https://api.idealista.com/3.5/"
    token_url: str = "https://api.idealista.com/oauth/token"
    usage_file: str = "api_usage.json"
    monthly_quota: int = 100
    min_days_between_similar_requests: int = 7

@dataclass
class SearchConfig:
    """Search-related configuration and parameters."""
    default_params: Dict
    city_coordinates: Dict
    city_location_ids: Dict
    max_items_per_page: int = 50
    max_pages: int = 20

# Search configuration
SEARCH_CONFIG = SearchConfig(
    default_params={
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
    },
    city_coordinates={
        "lisbon": "38.736946,-9.142685",
        "madrid": "40.416775,-3.703790",
        "barcelona": "41.385064,-2.173404",
    },
    city_location_ids={
        "lisbon": "0-EU-PT-11-06",
        "madrid": "0-EU-ES-28-07-001-079",
        "barcelona": "0-EU-ES-08-13-001-019",
    }
)

def load_config(env_file: Optional[str] = None) -> APIConfig:
    """
    Load API configuration from environment variables.
    
    Args:
        env_file: Optional path to .env file
        
    Returns:
        APIConfig: Configured API settings
        
    Raises:
        ValueError: If required credentials are missing
    """
    if env_file:
        # Clear existing environment variables to ensure clean state
        if "IDEALISTA_API_KEY" in os.environ:
            del os.environ["IDEALISTA_API_KEY"]
        if "IDEALISTA_CLIENT_SECRET" in os.environ:
            del os.environ["IDEALISTA_CLIENT_SECRET"]
        load_dotenv(env_file)
    else:
        load_dotenv()
        
    api_key = os.getenv("IDEALISTA_API_KEY")
    client_secret = os.getenv("IDEALISTA_CLIENT_SECRET")
    
    if not api_key or not client_secret:
        raise ValueError(
            "Missing required environment variables: 'IDEALISTA_API_KEY' or 'IDEALISTA_CLIENT_SECRET'"
        )
        
    return APIConfig(
        api_key=api_key,
        client_secret=client_secret
    )

def get_search_params(
    city: str = "lisbon",
    distance_to_center: Optional[int] = None,
    **overrides
) -> Dict:
    """
    Build search parameters for a specific city.
    
    Args:
        city: Name of the city (case-insensitive)
        distance_to_center: Optional distance in meters from city center
        **overrides: Additional parameters to override defaults
        
    Returns:
        Dict: Complete search parameters
        
    Raises:
        ValueError: If city is not configured or distance is invalid
    """
    city = city.lower()
    if city not in SEARCH_CONFIG.city_coordinates:
        raise ValueError(
            f"City '{city}' is not configured. Available cities: {list(SEARCH_CONFIG.city_coordinates.keys())}"
        )
        
    search_params = SEARCH_CONFIG.default_params.copy()
    
    if distance_to_center is not None:
        if distance_to_center <= 0:
            raise ValueError("distance_to_center must be a positive integer")
        search_params.update({
            "center": SEARCH_CONFIG.city_coordinates[city],
            "distance": distance_to_center,
        })
    else:
        search_params.update({
            "locationId": SEARCH_CONFIG.city_location_ids[city],
        })
        
    search_params.update(overrides)
    return search_params

# Default configuration (can be overridden in tests)
config = load_config()
