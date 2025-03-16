"""Tests for the configuration module."""

import os
import pytest
from pathlib import Path
from api_client.config import (
    APIConfig,
    SearchConfig,
    load_config,
    get_search_params,
    SEARCH_CONFIG
)

@pytest.fixture
def env_file(tmp_path):
    """Create a temporary .env file with test credentials."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "IDEALISTA_API_KEY=test_key\n"
        "IDEALISTA_CLIENT_SECRET=test_secret\n"
    )
    return str(env_path)

@pytest.fixture
def clean_env():
    """Ensure clean environment variables for each test."""
    # Save original environment
    original_env = {
        "IDEALISTA_API_KEY": os.environ.get("IDEALISTA_API_KEY"),
        "IDEALISTA_CLIENT_SECRET": os.environ.get("IDEALISTA_CLIENT_SECRET")
    }
    
    # Clear environment variables
    for key in original_env:
        if key in os.environ:
            del os.environ[key]
            
    yield
    
    # Restore original environment
    for key, value in original_env.items():
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]

def test_load_config_with_valid_env(env_file, clean_env):
    """Test loading configuration with valid environment variables."""
    config = load_config(env_file)
    
    assert isinstance(config, APIConfig)
    assert config.api_key == "test_key"
    assert config.client_secret == "test_secret"
    assert config.base_url == "https://api.idealista.com/3.5/"
    assert config.monthly_quota == 100

def test_load_config_missing_credentials(tmp_path, clean_env):
    """Test loading configuration with missing credentials."""
    empty_env = tmp_path / "empty.env"
    empty_env.touch()
    
    with pytest.raises(ValueError) as exc_info:
        load_config(str(empty_env))
    assert "Missing required environment variables" in str(exc_info.value)

def test_search_config_structure():
    """Test the structure of search configuration."""
    assert isinstance(SEARCH_CONFIG, SearchConfig)
    assert SEARCH_CONFIG.max_items_per_page == 50
    assert SEARCH_CONFIG.max_pages == 20
    
    # Test default parameters
    assert SEARCH_CONFIG.default_params["country"] == "pt"
    assert SEARCH_CONFIG.default_params["maxItems"] == "50"
    
    # Test city data
    assert "lisbon" in SEARCH_CONFIG.city_coordinates
    assert "madrid" in SEARCH_CONFIG.city_location_ids

def test_get_search_params_with_location_id():
    """Test getting search parameters using location ID."""
    params = get_search_params("lisbon")
    
    assert params["locationId"] == SEARCH_CONFIG.city_location_ids["lisbon"]
    assert "center" not in params
    assert "distance" not in params

def test_get_search_params_with_distance():
    """Test getting search parameters using distance to center."""
    params = get_search_params("madrid", distance_to_center=1000)
    
    assert params["center"] == SEARCH_CONFIG.city_coordinates["madrid"]
    assert params["distance"] == 1000
    assert "locationId" not in params

def test_get_search_params_with_overrides():
    """Test overriding default search parameters."""
    params = get_search_params(
        "barcelona",
        operation="sale",
        maxPrice="500000"
    )
    
    assert params["operation"] == "sale"
    assert params["maxPrice"] == "500000"
    assert params["locationId"] == SEARCH_CONFIG.city_location_ids["barcelona"]

def test_get_search_params_invalid_city():
    """Test getting search parameters for invalid city."""
    with pytest.raises(ValueError) as exc_info:
        get_search_params("invalid_city")
    assert "City 'invalid_city' is not configured" in str(exc_info.value)

def test_get_search_params_invalid_distance():
    """Test getting search parameters with invalid distance."""
    with pytest.raises(ValueError) as exc_info:
        get_search_params("lisbon", distance_to_center=-100)
    assert "distance_to_center must be a positive integer" in str(exc_info.value)

def test_case_insensitive_city_names():
    """Test that city names are case insensitive."""
    params1 = get_search_params("LISBON")
    params2 = get_search_params("lisbon")
    params3 = get_search_params("LiSbOn")
    
    assert params1 == params2 == params3 