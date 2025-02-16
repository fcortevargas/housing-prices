import os
import json
import base64
import logging
import datetime
import requests
from urllib.parse import urlencode, urljoin
from api_client.config import (
    API_KEY,
    CLIENT_SECRET,
    BASE_URL,
    TOKEN_URL,
    API_USAGE_FILE,
)


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

    def get_api_call_quota(self):
        """Return the current API call quota"""
        return self._usage_data["calls"]

    def get_last_api_search_date(self):
        """Return the last API search date."""
        return self._usage_data["last_api_search_date"]

    def define_search_url(self, params):
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

    def _update_api_call_count(self):
        self._usage_data["calls"] += 1
        self._save_api_usage()

    def update_last_api_search_date(self):
        """Update the last API search date to the current date."""
        self._usage_data["last_api_search_date"] = datetime.date.today().isoformat()
        self._save_api_usage()

    def search(self, params):
        """
        Executes a search API call using the provided parameters.

        :param params: A dictionary containing query parameters.
        :return: The API response as a JSON object.
        """
        self._validate_params(params)  # Validate parameters

        if self._usage_data["calls"] >= 100:
            logging.warning("API limit reached. Aborting search.")
            return None
        elif self._usage_data["last_api_search_date"] is not None:
            last_request_date = datetime.date.fromisoformat(
                self._usage_data["last_api_search_date"]
            )
            if datetime.date.today() - last_request_date < datetime.timedelta(
                days=self.API_CALL_INTERVAL
            ):
                logging.warning(
                    "Last API call executed less than a week ago. Aborting search."
                )
            return None

        url = self.define_search_url(params)
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = requests.post(url, headers=headers)

        if response.status_code == 200:
            self._update_api_call_count()
            return response.json()
        else:
            logging.error(f"API request failed: {response.text}")
            raise Exception("API request failed.")
