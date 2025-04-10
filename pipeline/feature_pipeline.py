"""
Feature engineering pipeline script.

This script handles the creation of all features for the housing price prediction model:
1. Location-based features (OSMNX)
2. Geodata features (proximity to POIs)
3. OpenAI-based text features
4. Derived numerical features
5. Feature unioning/joining

The script produces a feature-rich dataset ready for model training.
"""

import argparse
import logging
import os
import time
import dotenv
import yaml
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from src.preprocessors import (
    OpenAIFeatureExtractor,
    RelativeFeatureGenerator,
    ProximityScoreGenerator,
)

from src.utils.data_handlers import IdealistaDataSaver, IdealistaDataLoader

# Set up logging
from config.logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

# Load environment variables from .env file
dotenv.load_dotenv()


def generate_text_features(df, output_path, openai_api_key=None):
    """
    Generate features from property descriptions using OpenAI.

    Args:
        df: Input DataFrame with property listings
        output_path: Path to save the enhanced data
        openai_api_key: OpenAI API key (if None, tries to get from environment)

    Returns:
        DataFrame with added OpenAI features
    """
    logger.info("Starting text feature extraction with OpenAI")

    # Get API key from environment if not provided
    api_key = openai_api_key or os.getenv("OPEN_AI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided and not found in environment")

    # Create the feature extractor
    schema_path = (
        Path(__file__).resolve().parent.parent / "config" / "response.schema.json"
    )
    extractor = OpenAIFeatureExtractor(
        api_key=api_key, schema_path=str(schema_path), retry_attempts=3, delay_seconds=2
    )

    # Apply the extractor
    df_with_features = extractor.fit_transform(df)

    logger.info(
        f"Added {df_with_features.shape[1] - df.shape[1]} new features from text descriptions"
    )

    # Save the intermediate result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_with_features.to_csv(output_path)

    logger.info(f"Saved text features to {output_path}")

    return df_with_features


def generate_geodata_features(df, poi_geojson_dir, output_path):
    """
    Generate features based on proximity to points of interest.

    Args:
        df: Input DataFrame with property listings including lat/long
        poi_geojson_dir: Directory containing POI GeoJSON files
        output_path: Path to save the enhanced data

    Returns:
        DataFrame with added proximity features
    """
    logger.info("Starting geodata feature generation")

    # Here you would implement the geodata processing that's in your notebook
    # For example, calculate distances to nearest POIs of each type

    # This is a simplified example - in reality you would:
    # 1. Load GeoJSON files for different POI types (parks, transit, etc.)
    # 2. Calculate distances from each property to the nearest POI of each type
    # 3. Create new features based on these distances

    # Example of adding proximity score features
    proximity_generator = ProximityScoreGenerator(
        variables=[
            {
                "target": [
                    "distance_to_park",
                    "distance_to_metro",
                    "distance_to_supermarket",
                ],
                "agg": "mean",
                "name": "poi_proximity_score",
            }
        ]
    )

    # Only run this if these columns actually exist in the data
    if all(
        col in df.columns
        for col in ["distance_to_park", "distance_to_metro", "distance_to_supermarket"]
    ):
        df = proximity_generator.fit_transform(df)
    else:
        logger.warning(
            "Required distance columns not found, skipping proximity score generation"
        )

    # Save the intermediate result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)

    logger.info(f"Saved geodata features to {output_path}")

    return df


def generate_derived_features(df, output_path):
    """
    Generate derived features based on existing data.

    Args:
        df: Input DataFrame with property listings
        output_path: Path to save the enhanced data

    Returns:
        DataFrame with added derived features
    """
    logger.info("Starting derived feature generation")

    # Create price per square meter feature
    if "price" in df.columns and "size" in df.columns:
        df["price_per_sqm"] = df["price"] / df["size"]

    # Create relative features (e.g., price compared to neighborhood average)
    relative_generator = RelativeFeatureGenerator(
        variables=[
            {
                "groupby": "neighborhood",
                "target": "price",
                "agg": "mean",
                "name": "relative_price_to_neighborhood",
                "rescaling": True,
            }
        ]
    )

    # Only apply if neighborhood column exists
    if "neighborhood" in df.columns:
        df = relative_generator.fit_transform(df)

    # Save the intermediate result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)

    logger.info(f"Saved derived features to {output_path}")

    return df


def combine_features(config_file):
    """
    Combine all feature sets into one unified dataset.

    Args:
        feature_paths: Dictionary mapping feature types to file paths
        output_path: Path to save the combined dataset

    Returns:
        Combined DataFrame with all features
    """
    logger.info("Starting feature unioning process")

    with open(config_file) as f:
        raw_config = yaml.safe_load(f)

    config = raw_config.get("feature_combination")

    load_path = config.get("load_path")
    save_path = config.get("save_path")
    city = config.get("city")
    operation = config.get("operation")

    loader = IdealistaDataLoader(
        base_path=load_path,
        city=city,
        operation=operation,
    )

    saver = IdealistaDataSaver(
        base_path=save_path,
        city=city,
        operation=operation,
    )

    try:
        logging.info(f"Loading datasets from {load_path}")
        dataframes = loader.load_all_datasets()
        logger.info(f"Loaded {len(dataframes)} dataframes")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return

    # Combine the DataFrames
    combined_df = pd.concat(dataframes)

    # Keep only the last occurrence of each index
    combined_df = combined_df.loc[~combined_df.index.duplicated(keep="last")]

    # Keep only the listings that have unique values for the following columns
    combined_df = combined_df.loc[
        ~combined_df[
            [
                "floor",
                "propertyType",
                "size",
                "rooms",
                "bathrooms",
                "address",
                "description",
            ]
        ].duplicated(keep="last")
    ]

    # Filter by date
    from_date_str = config.get("from_date")
    to_date_str = config.get("to_date")

    if from_date_str is None:
        from_date = combined_df["snapshotDate"].min()
        from_date_str = from_date.strftime("%Y-%m-%d")
    if to_date_str is None:
        to_date = combined_df["snapshotDate"].max()
        to_date_str = to_date.strftime("%Y-%m-%d")
    else:
        to_date = datetime.datetime.strptime(to_date_str, "%Y-%m-%d").date()
        from_date = datetime.datetime.strptime(from_date_str, "%Y-%m-%d").date()
    if from_date > to_date:
        raise ValueError("from_date cannot be greater than to_date")

    combined_df = combined_df[combined_df["snapshotDate"] <= to_date]
    combined_df = combined_df[combined_df["snapshotDate"] >= from_date]

    # Save the combined DataFrame
    logger.info(f"Saving combined dataset to {save_path}")
    saver.set_file_name(from_date_str, to_date_str)
    saver.save_csv_data(combined_df)

    logger.info(
        f"Combining features from {from_date_str} to {to_date_str} completed successfully"
    )

    logger.info(
        f"Combined dataset has {combined_df.shape[1]} columns and {combined_df.shape[0]} rows"
    )

    return combined_df


def main():
    """Main function to run the feature engineering pipeline."""
    parser = argparse.ArgumentParser(
        description="Idealista feature engineering pipeline"
    )
    parser.add_argument(
        "--combine-features",
        action="store_true",
        help="Combine features from the datasets defined in the config file",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to feature engineering configuration file",
    )

    args = parser.parse_args()

    # Load the configuration file
    if args.config is None:
        raise ValueError("Configuration file path is required")
    config_file = args.config

    # Combine features
    if args.combine_features:
        combine_features(config_file)

    logger.info("Feature engineering pipeline completed successfully")


if __name__ == "__main__":
    main()
