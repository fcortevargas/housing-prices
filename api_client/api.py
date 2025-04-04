"""
Idealista API client implementation.

This module provides a robust API client for interacting with the Idealista API,
including authentication, request handling, and rate limiting.
"""

import os
import base64
import logging
import datetime
import requests
import pandas as pd
from typing import Optional, Dict, Any, List
from urllib.parse import urlencode, urljoin
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from api_client.config import IdealistaAPIConfig
from api_client.usage_tracker import IdealistaAPIUsageTracker
from utils.data_handlers import IdealistaDataSaver

logger = logging.getLogger(__name__)


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
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the API client with retry settings and usage tracking.

        Args:
            config_file: Optional path to config file
        """

        self.config = IdealistaAPIConfig(config_file)

        self.token_url = self.config.get_api_config("token_url")
        self.base_url = self.config.get_api_config("base_url")
        self.max_retries = self.config.get_api_config("max_retries")
        self.max_pages = self.config.get_api_config("max_pages")
        self.min_days_between_similar_requests = self.config.get_api_config(
            "min_days_between_similar_requests"
        )

        self.usage_tracker = IdealistaAPIUsageTracker(
            usage_file=self.config.get_api_config("usage_file"),
            monthly_quota=self.config.get_api_config("monthly_quota"),
        )

        self.search_params = None
        self.__access_token = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def _get_access_token(self) -> str:
        """
        Fetch an access token using OAuth2 client credentials.

        Returns:
            str: The access token

        Raises:
            AuthenticationError: If authentication fails
        """
        credentials = (
            f'{os.getenv("IDEALISTA_API_KEY")}:{os.getenv("IDEALISTA_CLIENT_SECRET")}'
        )
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

    def _can_execute_search(self) -> bool:
        """
        Checks if a search API call can be executed based on quotas and previous calls.

        Returns:
            bool: True if search can be executed
        """
        if not self.usage_tracker.can_make_call():
            logging.warning("Monthly API quota exceeded. Aborting.")
            return False

        api_search_params_list = self.usage_tracker.get_api_search_params_list()
        if api_search_params_list:
            for api_search_params in api_search_params_list:
                # Check if this is a duplicate request within the time window
                if self._is_similar_request(
                    api_search_params, self.config.prepare_search_params()
                ):
                    last_request_date = datetime.date.fromisoformat(
                        api_search_params["search_date"]
                    )
                    days_since_last_request = (
                        datetime.date.today() - last_request_date
                    ).days

                    if days_since_last_request < self.min_days_between_similar_requests:
                        logging.warning(
                            f"Similar request made {days_since_last_request} days ago. "
                            f"Must wait {self.min_days_between_similar_requests} days between similar requests."
                        )
                        return False
        return True

    @staticmethod
    def _is_similar_request(
        last_params: Dict[str, Any], params: Dict[str, Any]
    ) -> bool:
        """
        Determines whether two requests are similar based on specific parameters.

        Args:
            last_params (Dict[str, Any]): Parameters of the last request.
            params (Dict[str, Any]): Parameters of the current request.

        Returns:
            bool: True if the requests are similar, False otherwise.
        """
        return (
            last_params.get("operation") == params.get("operation")
            and last_params.get("propertyType") == params.get("propertyType")
            and last_params.get("locationId") == params.get("locationId")
            and last_params.get("center") == params.get("center")
            and last_params.get("distance") == params.get("distance")
        )

    def _define_search_url(self):
        """
        Builds a validated and encoded search URL.

        Returns:
            str: A fully constructed and encoded search URL.
        """
        # Build search base URL
        search_base = urljoin(self.base_url, f"{self.search_params['country']}/search")
        # Encode remaining parameters
        query_string = urlencode(self.search_params)
        return f"{search_base}?{query_string}"

    def _update_search_params(self, key: str, value: Any) -> None:
        """
        Update search parameters.

        Args:
            key: Key to update
            value: Value to update
        """
        self.search_params.update({key: value})

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def _search(self) -> Optional[Dict[str, Any]]:
        """
        Execute a search API call with automatic retries.

        Returns:
            Optional[Dict[str, Any]]: API response or None if request fails

        Raises:
            QuotaExceededError: If API quota is exceeded
            APIError: For other API-related errors
        """
        if not self.__access_token:
            raise AuthenticationError("No access token available")

        url = self._define_search_url()
        headers = {"Authorization": f"Bearer {self.__access_token}"}

        try:
            response = requests.post(url=url, headers=headers)
            response.raise_for_status()

            self.usage_tracker.increment_call_count()

            return response.json()

        except requests.exceptions.HTTPError as e:
            logging.error(f"API request failed: {e.response.text}")
            return None

    def fetch_data(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Fetch property listings data from the API.

        Returns:
            List[Dict[str, Any]]: List of property listings data
        """
        if self._can_execute_search():
            self.__access_token = self._get_access_token()

        api_call_quota = self.usage_tracker.get_remaining_calls()
        self.search_params = self.config.prepare_search_params()

        response = self._search()
        self.usage_tracker.update_api_search_params_list(self.search_params)

        if not response:
            return []

        total_pages = response.get("totalPages", 0)
        results = response.get("elementList", [])

        logging.info(f"Total pages to search: {total_pages}")
        logging.info(f"Searching page 1 of {total_pages}...")

        # Fetch additional pages if available and within quota
        if self.max_pages:
            page_limit = min(api_call_quota, total_pages, self.max_pages) + 1
        else:
            page_limit = min(api_call_quota, total_pages) + 1

        for page in range(2, page_limit):
            logging.info(f"Searching page {page} of {total_pages}...")
            if not self.usage_tracker.can_make_call():
                logging.warning("API quota reached - stopping pagination")
                break

            self._update_search_params("numPage", page)
            page_response = self._search()
            if page_response:
                results.extend(page_response.get("elementList", []))

        logging.info(f"Done fetching data from {total_pages} pages!")
        logging.info(f"Total number of listings found: {len(results)}")

        return results
