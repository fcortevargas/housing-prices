import os
import subprocess


def download_kaggle_dataset(dataset_name, local_path):
    """
    Downloads a dataset from Kaggle to a specified local path.

    Args:
        dataset_name (str): Kaggle dataset name in the format 'owner/dataset-name'.
        local_path (str): The local path where the dataset will be downloaded.
    """
    # Ensure the Kaggle credentials are set up
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_path):
        raise FileNotFoundError(
            "Kaggle API token (kaggle.json) not found. "
            "Place it in ~/.kaggle/ or configure the path."
        )

    # Create the local directory if it doesn't exist
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    try:
        # Run the Kaggle download command
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset_name,
                "--path",
                local_path,
                "--unzip",
            ],
            check=True,
        )
        print(f"Dataset '{dataset_name}' downloaded successfully to '{local_path}'")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
