import datetime
import logging
from api_client.api import IdealistaAPIClient

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    # Initialize API client
    client = IdealistaAPIClient()

    # Define city, search operation, and number of pages to fetch
    city = "lisbon"
    operation = "sale"
    sinceDate = "W"

    # Example: Fetch data for Lisbon with default params
    listings = client.fetch_data_for_city(
        city=city,
        operation=operation,
        sinceDate="W",
    )

    # Convert results to DataFrames
    df = client.results_to_df(listings)

    # Save DataFrames to CSV
    logging.info(f"Saving {len(df)} listings to CSV file...")
    df.to_csv(
        f"data/idealista/raw/{operation}/{city}/{datetime.date.today()}-{city}-listings-for-{operation}.csv",
        index=False,
    )
    logging.info("Done!")
