import datetime
import pandas as pd
import logging
from api_client.api import IdealistaAPIClient
from api_client.config import get_search_params

logging.basicConfig(level=logging.INFO)


def fetch_data_for_city(client, city, total_pages=2, **search_overrides):
    """
    Fetch data for a specific city.

    Args:
        client (IdealistaAPIClient): The API client.
        city (str): The city to search (e.g., 'lisbon', 'madrid').
        total_pages (int): Number of pages to fetch.
        **search_overrides: Additional search parameters to override defaults.

    Returns:
        list: Combined results from all pages.
    """
    combined_results = []
    for page in range(1, total_pages + 1):
        params = get_search_params(city, numPage=page, **search_overrides)
        logging.info(f"Fetching data for {city} (page {page})...")
        response = client.search(params)
        combined_results.extend(response.get("elementList", []))
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
    pages = 15

    # Example: Fetch data for Lisbon with default params
    listings = fetch_data_for_city(
        client=client,
        city=city,
        operation=operation,
        total_pages=pages,
    )

    # Convert results to DataFrames
    df = results_to_df(listings)

    # Save DataFrames to CSV
    logging.info(f"Saving {len(df)} listings to CSV...")
    df.to_csv(
        f"data/idealista/raw/{datetime.date.today()}-{city}-listings-for-{operation}.csv",
        index=False,
    )
    logging.info("Done!")
