import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from filelock import FileLock


class IdealistaAPIUsageTracker:
    """
    Tracks API usage with thread-safe operations and monthly quota management.

    This class ensures safe concurrent access to the usage data file and
    provides methods to track and validate API usage within monthly quotas.
    """

    def __init__(self, usage_file: str, monthly_quota: int = 100):
        """
        Initialize the API usage tracker.

        Args:
            usage_file: Path to the JSON file storing usage data
            monthly_quota: Maximum number of API calls allowed per month
        """
        self.usage_file = Path(usage_file)
        self.monthly_quota = monthly_quota
        self.lock_file = Path(str(usage_file) + ".lock")
        self._initialize_usage_file()

    def _initialize_usage_file(self):
        """Ensure usage file exists with initial structure."""
        if not self.usage_file.exists():
            usage_data = {
                "month": datetime.datetime.now().month,
                "calls": 0,
                "last_api_call_params": None,
            }
        else:
            usage_data = self._load_data()
            current_month = datetime.datetime.now().month
            if usage_data["month"] != current_month:
                usage_data.update({"month": current_month, "calls": 0})
        self._save_data(usage_data)

    def _load_data(self) -> Dict[str, Any]:
        """Thread-safe loading of usage data."""
        with FileLock(self.lock_file):
            with open(self.usage_file, "r") as f:
                return json.load(f)

    def _save_data(self, data: Dict[str, Any]):
        """Thread-safe saving of usage data."""
        with FileLock(self.lock_file):
            with open(self.usage_file, "w") as f:
                json.dump(data, f, indent=2)

    def increment_call_count(self) -> int:
        """
        Increment the API call count for the current month.

        Returns:
            int: Updated call count for the current month
        """
        usage_data = self._load_data()
        current_month = datetime.datetime.now().month

        if usage_data["month"] != current_month:
            usage_data.update({"month": current_month, "calls": 1})
        else:
            usage_data["calls"] += 1

        self._save_data(usage_data)
        return usage_data["calls"]

    def get_remaining_calls(self) -> int:
        """
        Get remaining API calls for the current month.

        Returns:
            int: Number of remaining API calls
        """
        usage_data = self._load_data()
        current_month = datetime.datetime.now().month

        if usage_data["month"] != current_month:
            return self.monthly_quota

        return max(0, self.monthly_quota - usage_data["calls"])

    def can_make_call(self) -> bool:
        """
        Check if an API call can be made within the monthly quota.

        Returns:
            bool: True if a call can be made, False otherwise
        """
        return self.get_remaining_calls() > 0

    def update_last_call_params(self, params: Dict[str, Any]):
        """
        Update the parameters of the last API call.

        Args:
            params: Dictionary containing the API call parameters
        """
        usage_data = self._load_data()
        params["search_date"] = datetime.date.today().isoformat()
        usage_data["last_api_call_params"] = params
        self._save_data(usage_data)

    def get_last_call_params(self) -> Optional[Dict[str, Any]]:
        """
        Get the parameters of the last API call.

        Returns:
            Optional[Dict[str, Any]]: Parameters of the last API call or None
        """
        return self._load_data().get("last_api_call_params")
