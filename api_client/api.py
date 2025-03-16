"""
Idealista API client implementation.

This module provides a robust API client for interacting with the Idealista API,
including authentication, request handling, and rate limiting.
"""

import os
import json
import base64
import logging
import datetime
import requests
import pandas as pd
from typing import Optional, Dict, Any, List
from urllib.parse import urlencode, urljoin
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from api_client.config import config, SEARCH_CONFIG, get_search_params
from api_client.usage_tracker import APIUsageTracker

logging.basicConfig(level=logging.INFO)

class APIError(Exception):
    """Base exception for API-related errors."""
    pass

class AuthenticationError(APIError):
    """Raised when authentication fails."""
    pass

class QuotaExceededError(APIError):
    """Raised when API quota is exceeded."""
    pass

class IdealistaAPIClient:
    def __init__(self, max_retries: int = 3):
        """
        Initialize the API client with retry settings and usage tracking.
        
        Args:
            max_retries: Maximum number of retry attempts for failed API calls
        """
        self.token_url = config.token_url
        self.base_url = config.base_url
        self.max_retries = max_retries
        self.usage_tracker = APIUsageTracker(
            config.usage_file,
            monthly_quota=config.monthly_quota
        )
        self.access_token = None
        
        if self.usage_tracker.can_make_call():
            self.access_token = self._get_access_token()
        else:
            logging.warning("API quota exceeded for this month.")
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException)
    )
    def _get_access_token(self) -> str:
        """
        Fetch an access token using OAuth2 client credentials.
        
        Returns:
            str: The access token
            
        Raises:
            AuthenticationError: If authentication fails
        """
        credentials = f"{config.api_key}:{config.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        }
        
        data = {"grant_type": "client_credentials", "scope": "read"}
        
        try:
            response = requests.post(self.token_url, data=data, headers=headers)
            response.raise_for_status()
            
            logging.info("Successfully obtained access token.")
            return response.json()["access_token"]
            
        except requests.exceptions.HTTPError as e:
            logging.error(f"Authentication failed: {e.response.text}")
            raise AuthenticationError("Failed to obtain access token") from e
            
    def _validate_params(self, params: Dict[str, Any]):
        """
        Validates required parameters for the search API.
        
        Args:
            params: Dictionary of query parameters
            
        Raises:
            ValueError: If required parameters are missing
        """
        required = ["country", "operation", "propertyType", "sinceDate"]
        location_required = ["center", "distance"]  # Or use 'locationId'
        
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")
            
        if not (all(k in params for k in location_required) or "locationId" in params):
            raise ValueError("Either 'center + distance' or 'locationId' must be specified.")
            
    def _can_execute_search(self, params: Dict[str, Any]) -> bool:
        """
        Checks if a search API call can be executed based on quotas and previous calls.
        
        Args:
            params: Search parameters to check
            
        Returns:
            bool: True if search can be executed
        """
        if not self.usage_tracker.can_make_call():
            logging.warning("Monthly API quota exceeded.")
            return False
            
        last_params = self.usage_tracker.get_last_call_params()
        if last_params:
            # Check if this is a duplicate request within the time window
            if (
                last_params.get("operation") == params.get("operation")
                and last_params.get("propertyType") == params.get("propertyType")
                and last_params.get("locationId") == params.get("locationId")
            ):
                last_request_date = datetime.date.fromisoformat(last_params["search_date"])
                days_since_last_request = (datetime.date.today() - last_request_date).days
                
                if days_since_last_request < config.min_days_between_similar_requests:
                    logging.warning(
                        f"Similar request made {days_since_last_request} days ago. "
                        f"Must wait {config.min_days_between_similar_requests} days between similar requests."
                    )
                    return False
                    
        return True
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException)
    )
    def search(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute a search API call with automatic retries.
        
        Args:
            params: Search parameters
            
        Returns:
            Optional[Dict[str, Any]]: API response or None if request fails
            
        Raises:
            QuotaExceededError: If API quota is exceeded
            APIError: For other API-related errors
        """
        if not self.access_token:
            raise AuthenticationError("No access token available")
            
        self._validate_params(params)
        
        if not self._can_execute_search(params):
            return None
            
        # Remove trailing slash from base_url if present to avoid double slashes
        base_url = self.base_url.rstrip('/')
        url = f"{base_url}/{params['country']}/search"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            response = requests.post(f"{url}?{urlencode(params)}", headers=headers)
            response.raise_for_status()
            
            self.usage_tracker.increment_call_count()
            self.usage_tracker.update_last_call_params(params)
            
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Too Many Requests
                raise QuotaExceededError("API rate limit exceeded")
            logging.error(f"API request failed: {e.response.text}")
            raise APIError(f"API request failed: {e.response.text}") from e
            
    def fetch_data_for_city(
        self, 
        city: str,
        operation: str = "sale",
        sinceDate: str = "W",
        max_pages: Optional[int] = None,
        **search_overrides
    ) -> List[Dict[str, Any]]:
        """
        Fetch property listings for a specific city.
        
        Args:
            city: Name of the city
            operation: Type of operation ('sale' or 'rent')
            sinceDate: Time window for listings ('W' for week, 'M' for month)
            max_pages: Optional override for maximum pages to fetch
            **search_overrides: Additional search parameters
            
        Returns:
            List[Dict[str, Any]]: List of property listings
        """
        params = get_search_params(city, **search_overrides)
        params.update({
            "operation": operation,
            "sinceDate": sinceDate,
            "maxItems": str(SEARCH_CONFIG.max_items_per_page)
        })
        
        logging.info(f"API calls remaining this month: {self.usage_tracker.get_remaining_calls()}")
        
        response = self.search(params)
        if not response:
            return []
            
        total_pages = response.get("totalPages", 0)
        results = response.get("elementList", [])
        
        # Fetch additional pages if available and within quota
        page_limit = min(
            total_pages + 1,
            max_pages or SEARCH_CONFIG.max_pages
        )
        
        for page in range(2, page_limit):
            if not self.usage_tracker.can_make_call():
                logging.warning("API quota reached. Stopping pagination.")
                break
                
            params["numPage"] = page
            page_response = self.search(params)
            if page_response:
                results.extend(page_response.get("elementList", []))
                
        return results
        
    @staticmethod
    def results_to_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert API results to a pandas DataFrame.
        
        Args:
            results: List of property listings
            
        Returns:
            pd.DataFrame: DataFrame containing the listings
            
        Raises:
            ValueError: If results list is empty
        """
        if not results:
            raise ValueError("No results to convert to DataFrame")
        return pd.DataFrame.from_records(results)
