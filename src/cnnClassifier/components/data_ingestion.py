import os
import zipfile
import kaggle
from dotenv import load_dotenv
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig

load_dotenv()   # loads KAGGLE_USERNAME and KAGGLE_KEY from .env


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Downloads the dataset from Kaggle using the kaggle package.
        Reads KAGGLE_USERNAME and KAGGLE_KEY from the .env file.
        """
        try:
            dataset_slug = self.config.source_url          # "masoudnickparvar/brain-tumor-mri-dataset"
            zip_local_data_file = self.config.local_data_file
            download_dir = os.path.dirname(zip_local_data_file)

            os.makedirs(download_dir, exist_ok=True)

            kaggle.api.authenticate()   # picks up KAGGLE_USERNAME / KAGGLE_KEY from env

            logger.info(f"Downloading Kaggle dataset: {dataset_slug}")
            kaggle.api.dataset_download_files(
                dataset_slug,
                path=download_dir,
                unzip=False,
                quiet=False
            )

            # kaggle saves as <dataset-name>.zip — rename to match config path
            dataset_name = dataset_slug.split("/")[-1]
            downloaded_zip = os.path.join(download_dir, f"{dataset_name}.zip")
            if os.path.exists(downloaded_zip) and downloaded_zip != str(zip_local_data_file):
                os.rename(downloaded_zip, zip_local_data_file)

            logger.info(f"Dataset downloaded to: {zip_local_data_file}")

        except Exception as e:
            raise e

    def extract_file(self):
        """Extracts the downloaded ZIP into the unzip directory."""
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)

            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            logger.info(f"Dataset extracted to: {unzip_path}")

        except Exception as e:
            raise e