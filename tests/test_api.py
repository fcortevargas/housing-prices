"""Tests for the API client module."""

import pytest
import responses
import json
from datetime import datetime, date
from pathlib import Path
from urllib.parse import urlencode
from api_client.api import (
    IdealistaAPIClient,
    APIError,
    AuthenticationError,
    QuotaExceededError
)
from api_client.config import config

@pytest.fixture
def mock_token_response():
    """Mock successful token response."""
    return {
        "access_token": "mock_token",
        "token_type": "bearer",
        "expires_in": 43200
    }

@pytest.fixture
def mock_search_response():
    """Mock successful search response."""
    return {
        "elementList": [
            {"id": 1, "price": 100000},
            {"id": 2, "price": 200000}
        ],
        "total": 2,
        "totalPages": 1
    }

@pytest.fixture
def clean_usage_file(tmp_path):
    """Create a clean usage file for each test."""
    usage_file = tmp_path / "api_usage.json"
    initial_data = {
        "month": datetime.now().month,
        "calls": 0,
        "last_api_call_params": None
    }
    usage_file.write_text(json.dumps(initial_data))
    return str(usage_file)

@pytest.fixture
def client(clean_usage_file, monkeypatch, mock_token_response):
    """Create an API client with a clean usage tracker."""
    monkeypatch.setenv("IDEALISTA_API_KEY", "test_key")
    monkeypatch.setenv("IDEALISTA_CLIENT_SECRET", "test_secret")
    
    # Mock the config
    mock_config = config
    mock_config.usage_file = clean_usage_file
    monkeypatch.setattr("api_client.api.config", mock_config)
    
    # Set up mock token response
    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            config.token_url,
            json=mock_token_response,
            status=200
        )
        client = IdealistaAPIClient()
        
    return client

@responses.activate
def test_authentication_success(mock_token_response):
    """Test successful API authentication."""
    # Mock the token endpoint
    responses.add(
        responses.POST,
        config.token_url,
        json=mock_token_response,
        status=200
    )
    
    client = IdealistaAPIClient()
    assert client.access_token == "mock_token"

@responses.activate
def test_authentication_failure(clean_usage_file, monkeypatch):
    """Test authentication failure handling."""
    monkeypatch.setenv("IDEALISTA_API_KEY", "test_key")
    monkeypatch.setenv("IDEALISTA_CLIENT_SECRET", "test_secret")
    
    # Mock the config
    mock_config = config
    mock_config.usage_file = clean_usage_file
    monkeypatch.setattr("api_client.api.config", mock_config)
    
    # Mock failed authentication
    responses.add(
        responses.POST,
        config.token_url,
        json={"error": "invalid_client"},
        status=401
    )
    
    with pytest.raises(AuthenticationError):
        IdealistaAPIClient()

@responses.activate
def test_search_success(mock_token_response, mock_search_response, client):
    """Test successful property search."""
    params = {
        "country": "pt",
        "operation": "sale",
        "propertyType": "homes",
        "locationId": "0-EU-PT-11-06",
        "sinceDate": "W"
    }
    
    # Mock the search endpoint with query parameters
    base_url = client.base_url.rstrip('/')
    url = f"{base_url}/{params['country']}/search?{urlencode(params)}"
    responses.add(
        responses.POST,
        url,
        json=mock_search_response,
        status=200
    )
    
    response = client.search(params)
    assert response["elementList"][0]["price"] == 100000
    assert len(response["elementList"]) == 2

@responses.activate
def test_quota_exceeded(mock_token_response, client):
    """Test handling of exceeded API quota."""
    params = {
        "country": "pt",
        "operation": "rent",  # Different operation to avoid duplicate request check
        "propertyType": "homes",
        "locationId": "0-EU-PT-11-06",
        "sinceDate": "W"
    }
    
    # Mock the quota exceeded response with query parameters
    base_url = client.base_url.rstrip('/')
    url = f"{base_url}/{params['country']}/search?{urlencode(params)}"
    responses.add(
        responses.POST,
        url,
        json={"error": "quota_exceeded"},
        status=429
    )
    
    with pytest.raises(QuotaExceededError):
        client.search(params)

def test_validate_params(client):
    """Test parameter validation."""
    # Test missing required params
    with pytest.raises(ValueError) as exc_info:
        client._validate_params({})
    assert "Missing required parameters" in str(exc_info.value)
    
    # Test missing location info
    params = {
        "country": "pt",
        "operation": "sale",
        "propertyType": "homes",
        "sinceDate": "W"
    }
    with pytest.raises(ValueError) as exc_info:
        client._validate_params(params)
    assert "center + distance" in str(exc_info.value)

@responses.activate
def test_fetch_city_data(mock_token_response, mock_search_response, client):
    """Test fetching data for a city."""
    # First page parameters
    first_page_params = {
        "country": "pt",
        "locale": "pt",
        "language": "pt",
        "maxItems": "50",
        "operation": "rent",  # Different operation to avoid duplicate request check
        "propertyType": "homes",
        "order": "publicationDate",
        "sort": "desc",
        "maxPrice": "100000000",
        "sinceDate": "W",
        "locationId": "0-EU-PT-11-06"
    }
    
    # Mock first page response
    base_url = client.base_url.rstrip('/')
    first_url = f"{base_url}/{first_page_params['country']}/search?{urlencode(first_page_params)}"
    responses.add(
        responses.POST,
        first_url,
        json=mock_search_response,
        status=200
    )
    
    # Second page parameters
    second_page_params = first_page_params.copy()
    second_page_params["numPage"] = 2
    
    # Mock second page response
    second_url = f"{base_url}/{second_page_params['country']}/search?{urlencode(second_page_params)}"
    responses.add(
        responses.POST,
        second_url,
        json=mock_search_response,
        status=200
    )
    
    results = client.fetch_data_for_city("lisbon", operation="rent")
    assert len(results) == 2
    assert results[0]["price"] == 100000