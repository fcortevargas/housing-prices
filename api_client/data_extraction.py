"""
Data extraction module for the Idealista API.

This module handles the extraction and storage of property listings
from the Idealista API, with proper error handling and data persistence.
"""

import logging
from typing import Optional, List
from api_client.api import IdealistaAPIClient
from api_client.storage import DataStorage
from api_client.config import config, SEARCH_CONFIG

logging.basicConfig(level=logging.INFO)

def extract_city_data(
    city: str,
    operation: str = "sale",
    since_date: str = "W",
    max_pages: Optional[int] = None
) -> bool:
    """
    Extract and safely store data for a city.
    
    Args:
        city: Name of the city to extract data for
        operation: Type of operation ('sale' or 'rent')
        since_date: Time window for listings ('W' for week, 'M' for month)
        max_pages: Optional override for maximum pages to fetch
        
    Returns:
        bool: True if extraction and storage was successful
    """
    # Initialize API client and storage
    client = IdealistaAPIClient()
    storage = DataStorage()
    
    try:
        # Fetch listings with configurable page limit
        listings = client.fetch_data_for_city(
            city=city,
            operation=operation,
            sinceDate=since_date,
            maxItems=str(SEARCH_CONFIG.max_items_per_page),
            max_pages=max_pages or SEARCH_CONFIG.max_pages
        )
        
        if not listings:
            logging.warning(f"No listings found for {city}")
            return False
            
        # Save raw data first
        if not storage.save_raw_listings(listings, city, operation):
            logging.error("Failed to save raw listings")
            return False
            
        # Process and save as DataFrame
        df = client.results_to_df(listings)
        if storage.save_processed_listings(df, city, operation):
            logging.info(f"Successfully saved {len(df)} listings")
            return True
        else:
            logging.error("Failed to save processed listings")
            return False
            
    except Exception as e:
        logging.error(f"Error extracting data for {city}: {e}")
        return False

def main():
    """Main entry point for data extraction."""
    # Cities to extract data from
    cities = ["lisbon"]
    
    # Extract data for each city with default parameters
    for city in cities:
        if extract_city_data(
            city=city,
            operation="sale",
            since_date="W",
            max_pages=None
        ):
            logging.info(f"Successfully processed {city} data")
        else:
            logging.error(f"Failed to process {city} data")

if __name__ == "__main__":
    main()
