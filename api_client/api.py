import requests
import base64
import logging
from urllib.parse import urlencode, urljoin
from api_client.config import API_KEY, CLIENT_SECRET, BASE_URL, TOKEN_URL

logging.basicConfig(level=logging.INFO)


class IdealistaAPIClient:
    def __init__(self):
        self.token_url = TOKEN_URL
        self.access_token = self._get_access_token()
        self.base_url = BASE_URL

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
        required = ["country", "operation", "propertyType"]
        location_required = ["center", "distance"]  # Or use 'locationId'
        missing = [key for key in required if key not in params]

        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")

        # Ensure either 'center + distance' or 'locationId' is present
        if not (all(k in params for k in location_required) or "locationId" in params):
            raise ValueError(
                "Either 'center + distance' or 'locationId' must be specified."
            )

    def define_search_url(self, params):
        """
        Builds a validated and encoded search URL.

        :param params: A dictionary containing query parameters.
        :return: A fully constructed and encoded search URL.
        """
        self._validate_params(params)  # Validate parameters
        search_base = urljoin(
            self.base_url, f"{params['country']}/search"
        )  # Build search base URL
        query_string = urlencode(params)  # Encode remaining parameters
        return f"{search_base}?{query_string}"

    def search(self, params):
        """
        Executes a search API call using the provided parameters.

        :param params: A dictionary containing query parameters.
        :return: The API response as a JSON object.
        """
        url = self.define_search_url(params)
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = requests.post(url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"API request failed: {response.text}")
            raise Exception("API request failed.")
