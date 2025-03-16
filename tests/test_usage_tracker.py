import pytest
import json
from datetime import datetime
from pathlib import Path
from api_client.usage_tracker import APIUsageTracker

@pytest.fixture
def temp_usage_file(tmp_path):
    """Create a temporary usage file for testing."""
    usage_file = tmp_path / "api_usage.json"
    return str(usage_file)

@pytest.fixture
def tracker(temp_usage_file):
    """Create an APIUsageTracker instance with a temporary file."""
    return APIUsageTracker(usage_file=temp_usage_file, monthly_quota=100)

def test_initial_state(tracker, temp_usage_file):
    """Test initial state of the usage tracker."""
    assert Path(temp_usage_file).exists()
    data = json.loads(Path(temp_usage_file).read_text())
    assert data["month"] == datetime.now().month
    assert data["calls"] == 0
    assert data["last_api_call_params"] is None

def test_increment_call_count(tracker):
    """Test incrementing call count."""
    assert tracker.increment_call_count() == 1
    assert tracker.increment_call_count() == 2
    assert tracker._load_data()["calls"] == 2

def test_remaining_calls(tracker):
    """Test remaining calls calculation."""
    initial_remaining = tracker.get_remaining_calls()
    assert initial_remaining == 100
    
    tracker.increment_call_count()
    assert tracker.get_remaining_calls() == 99

def test_can_make_call(tracker):
    """Test call permission check."""
    assert tracker.can_make_call() is True
    
    # Simulate reaching the quota
    for _ in range(100):
        tracker.increment_call_count()
    
    assert tracker.can_make_call() is False

def test_update_last_call_params(tracker):
    """Test updating and retrieving last call parameters."""
    test_params = {
        "city": "lisbon",
        "operation": "sale"
    }
    
    tracker.update_last_call_params(test_params)
    last_params = tracker.get_last_call_params()
    
    assert last_params["city"] == "lisbon"
    assert last_params["operation"] == "sale"
    assert "search_date" in last_params

def test_month_rollover(tracker):
    """Test behavior when month changes."""
    # Set up initial state
    tracker.increment_call_count()
    data = tracker._load_data()
    
    # Simulate month change by directly modifying the data
    data["month"] = (datetime.now().month - 1) % 12 or 12  # Handle December rollover
    tracker._save_data(data)
    
    # Next increment should reset the counter
    assert tracker.increment_call_count() == 1 