from src.kaggle import download_kaggle_dataset

if __name__ == "__main__":
    # Download the dataset
    dataset = "victorianomh/idealista-barcelona-raw-scraped-data"
    path = f"./data/{dataset}"
    download_kaggle_dataset(dataset, path)
