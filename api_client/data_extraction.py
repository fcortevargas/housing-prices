import datetime
import pandas as pd
import logging
from api_client.api import IdealistaAPIClient
from api_client.config import get_search_params

logging.basicConfig(level=logging.INFO)


def fetch_data_for_city(client: IdealistaAPIClient, city: str, **search_overrides):
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

    logging.info(f"API calls made this month: {client.get_api_call_quota()} / 100.")
    logging.info(f"Last API search request: {client.get_last_api_search_date()}.")

    # Fetch first page to determine total pages
    logging.info(f"Fetching page 1...")
    response = client.search(params)

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

    api_call_quota = client.get_api_call_quota()
    logging.info(f"The current API call quota is {api_call_quota}.")

    pagination_count = min(total_pages, 100 - api_call_quota)
    logging.info(f"Will fetch a total of {pagination_count} pages.")

    for page in range(2, pagination_count + 1):
        params["numPage"] = page
        logging.info(f"Fetching page {page} of {pagination_count}...")
        response = client.search(params)
        if not response:
            break
        combined_results.extend(response.get("elementList", []))

    client.update_last_api_search_date()

    return combined_results


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


if __name__ == "__main__":
    # Initialize API client
    client = IdealistaAPIClient()

    # Define city, search operation, and number of pages to fetch
    city = "lisbon"
    operation = "rent"

    # Example: Fetch data for Lisbon with default params
    listings = fetch_data_for_city(
        client=client,
        city=city,
        operation=operation,
    )

    # Convert results to DataFrames
    df = results_to_df(listings)

    # Save DataFrames to CSV
    logging.info(f"Saving {len(df)} listings to CSV file...")
    df.to_csv(
        f"data/idealista/raw/{operation}/{city}/{datetime.date.today()}-{city}-listings-for-{operation}.csv",
        index=False,
    )
    logging.info("Done!")
