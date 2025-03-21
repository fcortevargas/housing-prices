import pytest
import pandas as pd
from pathlib import Path
import shutil
from api_client.data_storage import DataStorage

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path

@pytest.fixture
def storage(temp_data_dir):
    """Create a DataStorage instance with a temporary directory."""
    return DataStorage(base_path=str(temp_data_dir))

def test_directory_creation(storage, temp_data_dir):
    """Test that all necessary directories are created."""
    assert (temp_data_dir / "raw").exists()
    assert (temp_data_dir / "interim").exists()
    assert (temp_data_dir / "processed").exists()

def test_save_raw_listings(storage, temp_data_dir):
    """Test saving raw listings data."""
    # Test data
    listings = [
        {"id": 1, "price": 100000, "location": "Test Location"},
        {"id": 2, "price": 200000, "location": "Another Location"}
    ]
    
    # Save the data
    result = storage.save_raw_listings(listings, "test_city", "sale")
    
    # Check results
    assert result is True
    saved_files = list((temp_data_dir / "raw").glob("*.json"))
    assert len(saved_files) == 1
    
def test_save_processed_listings(storage, temp_data_dir):
    """Test saving processed listings data."""
    # Test data
    df = pd.DataFrame({
        "id": [1, 2],
        "price": [100000, 200000],
        "location": ["Test Location", "Another Location"]
    })
    
    # Save the data
    result = storage.save_processed_listings(df, "test_city", "sale")
    
    # Check results
    assert result is True
    saved_files = list((temp_data_dir / "processed").glob("*.csv"))
    assert len(saved_files) == 1

def test_save_empty_data(storage):
    """Test handling of empty data."""
    # Empty list for raw data
    assert storage.save_raw_listings([], "test_city", "sale") is False
    
    # Empty DataFrame for processed data
    empty_df = pd.DataFrame()
    assert storage.save_processed_listings(empty_df, "test_city", "sale") is False 