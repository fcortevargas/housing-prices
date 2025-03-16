"""Tests for the data extraction module."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, call
from api_client.data_extraction import extract_city_data
from api_client.config import SEARCH_CONFIG

@pytest.fixture
def mock_listings():
    """Sample listings data for testing."""
    return [
        {"id": 1, "price": 100000, "location": "Test Location 1"},
        {"id": 2, "price": 200000, "location": "Test Location 2"}
    ]

@pytest.fixture
def mock_api_client(mock_listings):
    """Mock API client for testing."""
    mock_client = Mock()
    mock_client.fetch_data_for_city.return_value = mock_listings
    mock_client.results_to_df.return_value = pd.DataFrame(mock_listings)
    return mock_client

@pytest.fixture
def mock_storage():
    """Mock storage for testing."""
    mock_storage = Mock()
    mock_storage.save_raw_listings.return_value = True
    mock_storage.save_processed_listings.return_value = True
    return mock_storage

@patch('api_client.data_extraction.IdealistaAPIClient')
@patch('api_client.data_extraction.DataStorage')
def test_successful_extraction(MockStorage, MockAPIClient, mock_api_client, mock_storage, mock_listings):
    """Test successful data extraction and storage."""
    # Set up mocks
    MockAPIClient.return_value = mock_api_client
    MockStorage.return_value = mock_storage
    
    # Execute extraction
    result = extract_city_data("lisbon", operation="sale")
    
    # Verify success
    assert result is True
    
    # Verify API client calls
    mock_api_client.fetch_data_for_city.assert_called_once_with(
        city="lisbon",
        operation="sale",
        sinceDate="W",
        maxItems=str(SEARCH_CONFIG.max_items_per_page),
        max_pages=SEARCH_CONFIG.max_pages
    )
    mock_api_client.results_to_df.assert_called_once_with(mock_listings)
    
    # Verify storage calls
    mock_storage.save_raw_listings.assert_called_once_with(
        mock_listings, "lisbon", "sale"
    )
    mock_storage.save_processed_listings.assert_called_once()

@patch('api_client.data_extraction.IdealistaAPIClient')
@patch('api_client.data_extraction.DataStorage')
def test_extraction_with_custom_pages(MockStorage, MockAPIClient, mock_api_client, mock_storage):
    """Test extraction with custom max_pages parameter."""
    MockAPIClient.return_value = mock_api_client
    MockStorage.return_value = mock_storage
    
    # Execute extraction with custom pages
    extract_city_data("madrid", max_pages=5)
    
    # Verify custom pages parameter was passed
    mock_api_client.fetch_data_for_city.assert_called_once_with(
        city="madrid",
        operation="sale",
        sinceDate="W",
        maxItems=str(SEARCH_CONFIG.max_items_per_page),
        max_pages=5
    )

@patch('api_client.data_extraction.IdealistaAPIClient')
@patch('api_client.data_extraction.DataStorage')
def test_extraction_no_listings(MockStorage, MockAPIClient, mock_api_client, mock_storage):
    """Test handling of no listings found."""
    # Configure mock to return no listings
    mock_api_client.fetch_data_for_city.return_value = []
    MockAPIClient.return_value = mock_api_client
    MockStorage.return_value = mock_storage
    
    # Execute extraction
    result = extract_city_data("barcelona")
    
    # Verify failure
    assert result is False
    
    # Verify storage was not called
    mock_storage.save_raw_listings.assert_not_called()
    mock_storage.save_processed_listings.assert_not_called()

@patch('api_client.data_extraction.IdealistaAPIClient')
@patch('api_client.data_extraction.DataStorage')
def test_extraction_raw_storage_failure(MockStorage, MockAPIClient, mock_api_client, mock_storage, mock_listings):
    """Test handling of raw storage failure."""
    # Configure mock for storage failure
    mock_storage.save_raw_listings.return_value = False
    MockAPIClient.return_value = mock_api_client
    MockStorage.return_value = mock_storage
    
    # Execute extraction
    result = extract_city_data("lisbon")
    
    # Verify failure
    assert result is False
    
    # Verify processed storage was not attempted
    mock_storage.save_processed_listings.assert_not_called()

@patch('api_client.data_extraction.IdealistaAPIClient')
@patch('api_client.data_extraction.DataStorage')
def test_extraction_processed_storage_failure(MockStorage, MockAPIClient, mock_api_client, mock_storage, mock_listings):
    """Test handling of processed storage failure."""
    # Configure mock for processed storage failure
    mock_storage.save_processed_listings.return_value = False
    MockAPIClient.return_value = mock_api_client
    MockStorage.return_value = mock_storage
    
    # Execute extraction
    result = extract_city_data("lisbon")
    
    # Verify failure
    assert result is False

@patch('api_client.data_extraction.IdealistaAPIClient')
@patch('api_client.data_extraction.DataStorage')
def test_extraction_api_error(MockStorage, MockAPIClient, mock_storage):
    """Test handling of API errors."""
    # Configure mock to raise an exception
    mock_client = Mock()
    mock_client.fetch_data_for_city.side_effect = Exception("API Error")
    MockAPIClient.return_value = mock_client
    MockStorage.return_value = mock_storage
    
    # Execute extraction
    result = extract_city_data("lisbon")
    
    # Verify failure
    assert result is False
    
    # Verify storage was not called
    mock_storage.save_raw_listings.assert_not_called()
    mock_storage.save_processed_listings.assert_not_called()

@patch('api_client.data_extraction.extract_city_data')
def test_main_function(mock_extract):
    """Test the main function."""
    from api_client.data_extraction import main
    
    # Configure mock responses for different cities
    mock_extract.side_effect = [True, False, True]  # Success, failure, success
    
    # Execute main function
    main()
    
    # Verify all cities were processed
    assert mock_extract.call_count == len(SEARCH_CONFIG.city_location_ids)
    
    # Verify cities were processed with default parameters
    expected_calls = [
        call(city=city, operation="sale", since_date="W", max_pages=None)
        for city in SEARCH_CONFIG.city_location_ids.keys()
    ]
    mock_extract.assert_has_calls(expected_calls) 