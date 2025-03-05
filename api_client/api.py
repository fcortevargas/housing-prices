import os
import json
import base64
import logging
import datetime
import requests
import pandas as pd
from urllib.parse import urlencode, urljoin
from api_client.config import (
    API_KEY,
    CLIENT_SECRET,
    BASE_URL,
    TOKEN_URL,
    API_USAGE_FILE,
)
from api_client.config import get_search_params


logging.basicConfig(level=logging.INFO)


class IdealistaAPIClient:
    def __init__(self):
        self.token_url = TOKEN_URL
        self.base_url = BASE_URL
        self._usage_data = self._load_api_usage()
        self.API_CALL_INTERVAL = 7

        if self._usage_data["calls"] >= 100:
            logging.warning("API limit reached for the month. Stopping execution.")
            self.access_token = None
        else:
            self.access_token = self._get_access_token()

    def _get_access_token(self):
        """
        Fetches an access token using OAuth2 client credentials.
        """
        credentials = f"{API_KEY}:{CLIENT_SECRET}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        }
        data = {"grant_type": "client_credentials", "scope": "read"}
        response = requests.post(self.token_url, data=data, headers=headers)

        if response.status_code == 200:
            logging.info("Successfully obtained access token.")
            self._update_api_call_count()
            return response.json()["access_token"]
        else:
            logging.error(f"Failed to obtain access token: {response.text}")
            raise Exception("Authorization failed.")

    def _validate_params(self, params):
        """
        Validates required parameters for the search API.

        :param params: A dictionary containing query parameters.
        :raises ValueError: If required parameters are missing.
        """
        required = ["country", "operation", "propertyType", "sinceDate"]
        location_required = ["center", "distance"]  # Or use 'locationId'
        missing = [key for key in required if key not in params]

        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")

        # Ensure either 'center + distance' or 'locationId' is present
        if not (all(k in params for k in location_required) or "locationId" in params):
            raise ValueError(
                "Either 'center + distance' or 'locationId' must be specified."
            )

        if params["sinceDate"] == "W":
            self.API_CALL_INTERVAL = 7
        elif params["sinceDate"] == "M":
            self.API_CALL_INTERVAL = 30
        else:
            raise ValueError("'sinceDate' must be 'W' or 'M'.")

    def _define_search_url(self, params):
        """
        Builds a validated and encoded search URL.

        :param params: A dictionary containing query parameters.
        :return: A fully constructed and encoded search URL.
        """
        search_base = urljoin(
            self.base_url, f"{params['country']}/search"
        )  # Build search base URL
        query_string = urlencode(params)  # Encode remaining parameters
        return f"{search_base}?{query_string}"

    def _load_api_usage(self):
        """Loads or initializes API usage tracking."""
        if os.path.exists(API_USAGE_FILE):
            with open(API_USAGE_FILE, "r") as file:
                data = json.load(file)
        else:
            data = {
                "month": datetime.datetime.now().month,
                "calls": 0,
                "last_api_search_date": None,
            }

        if data["month"] != datetime.datetime.now().month:
            data.update(
                {"month": datetime.datetime.now().month, "calls": 0}
            )  # Reset count

        return data

    def _save_api_usage(self):
        """
        Saves the updated data to the API usage file.
        """
        with open(API_USAGE_FILE, "w") as file:
            json.dump(self._usage_data, file)

    def _update_api_call_count(self):
        self._usage_data["calls"] += 1

    def _update_last_api_search_parameters(self, params):
        """Update the last API search parameters."""
        self._usage_data["last_api_call_params"] = params
        self._usage_data["last_api_call_params"][
            "search_date"
        ] = datetime.date.today().isoformat()
        self._save_api_usage()

    def get_api_call_quota(self):
        """Return the current API call quota"""
        return self._usage_data["calls"]

    def get_last_api_search_date(self):
        """Return the last API search date."""
        return self._usage_data["last_api_call_params"].get("search_date")

    def _can_execute_search(self, params):
        """
        Checks if a search API call can be executed based on the current API usage.

        :return: True if a search can be executed, False otherwise.
        """
        if self._usage_data["calls"] >= 100:
            logging.warning("API limit reached. Aborting search.")
            return False
        if self._usage_data["last_api_call_params"].get("search_date") is not None:
            if (
                self._usage_data["last_api_call_params"].get("operation")
                == params.get("operation")
                and self._usage_data["last_api_call_params"].get("propertyType")
                == params.get("propertyType")
                and self._usage_data["last_api_call_params"].get("locationId")
                == params.get("locationId")
            ):
                last_request_date = datetime.date.fromisoformat(
                    self._usage_data["last_api_call_params"].get("search_date")
                )
                if datetime.date.today() - last_request_date < datetime.timedelta(
                    days=self.API_CALL_INTERVAL
                ):
                    warning_message = f"The last API call for the given search parameters was executed less {self.API_CALL_INTERVAL} ago. Aborting search."
                    logging.warning(warning_message)
                    return False
        return True

    def search(self, params):
        """
        Executes a search API call using the provided parameters.

        :param params: A dictionary containing query parameters.
        :return: The API response as a JSON object.
        """
        self._validate_params(params)  # Validate parameters

        if not self._can_execute_search(params):
            return None

        url = self._define_search_url(params)
        headers = {"Authorization": f"Bearer {self.access_token}"}
        try:
            response = requests.post(url, headers=headers)
            self._update_api_call_count()
        except Exception as e:
            logging.error(f"Error: {e}")

        if response.status_code == 200:
            logging.info("API request successful.")
            return response.json()
        else:
            logging.error(f"API request failed: {response.text}")
            raise Exception("API request failed.")

    def fetch_data_for_city(self, city: str, **search_overrides):
        """
        Fetch data for a specific city dynamically until all pages are retrieved or API calls run out.

        Args:
            client (IdealistaAPIClient): The API client.
            city (str): The city to search (e.g., 'lisbon', 'madrid').
            **search_overrides: Additional search parameters to override defaults.

        Returns:
            list: Combined results from all pages.
        """
        params = get_search_params(city, **search_overrides)

        logging.info(f"API calls made this month: {self.get_api_call_quota()} / 100.")

        # Fetch first page to determine total pages
        logging.info(f"Fetching page 1...")
        response = self.search(params)

        if not response:
            logging.warning(f"Response is empty.")
            return []

        summary = response.get("summary")
        logging.info(f"Request summary: {summary}")

        total = response.get("total")
        logging.info(f"Found {total} total listings for {city}.")

        total_pages = response.get("totalPages")
        logging.info(f"Found {total_pages} total pages to be fetched.")
        combined_results = response.get("elementList", [])

        api_call_quota = self.get_api_call_quota()
        logging.info(f"The current API call quota is {api_call_quota}.")

        pagination_count = min(total_pages, 100 - api_call_quota)
        logging.info(f"Will fetch a total of {pagination_count} pages.")

        for page in range(2, pagination_count + 1):
            params["numPage"] = page
            logging.info(f"Fetching page {page} of {pagination_count}...")
            response = self.search(params)
            if not response:
                break
            combined_results.extend(response.get("elementList", []))

        self._update_last_api_search_parameters(params)

        return combined_results

    @staticmethod
    def results_to_df(results):
        """
        Converts a list of results into a Pandas DataFrame.

        Args:
            results (list): List of dictionaries representing the search results.

        Returns:
            pd.DataFrame: DataFrame containing the search results.
        """
        if not results:
            raise ValueError("No results found or invalid input.")
        return pd.DataFrame.from_records(results)
