
import os
import shutil
import zipfile
import requests
import anndata
from tqdm import tqdm

class PertDataset:
    """Dataset class for perturbation data."""

    def __init__(self, name: str, cache_dir_path: str = "data", silent: bool = False):
        """Initialize the PertDataset.

        Args:
            name: Name of the dataset (e.g., "norman").
            cache_dir_path: Path to the directory where the data should be cached.
            silent: Whether to suppress output.
        """
        self.name = name
        self.cache_dir_path = os.path.abspath(cache_dir_path)
        self.silent = silent
        self.path = os.path.join(self.cache_dir_path, name)
        
        # Define dataset URLs
        if name == "norman":
            self.url = "https://dataverse.harvard.edu/api/access/datafile/6154020"
        else:
            raise ValueError(f"Unknown dataset: {name}")

        self._download_and_extract()
        self.adata = self._load_data()

    def _download_and_extract(self):
        """Download and extract the dataset."""
        if os.path.exists(self.path):
            if not self.silent:
                print(f"Dataset already downloaded at {self.path}")
            return

        os.makedirs(self.path, exist_ok=True)
        zip_path = os.path.join(self.path, "data.zip")

        if not self.silent:
            print(f"Downloading: {self.url} -> {zip_path}")

        response = requests.get(self.url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        if not self.silent:
             print(f"Total size: {total_size_in_bytes:,} bytes")

        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, disable=self.silent)
        
        with open(zip_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if not self.silent:
            print(f"Download completed: {zip_path}")
            print(f"Extracting to: {self.path}")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.path)

    def _load_data(self):
        """Load the data into an AnnData object."""
        # Find the h5ad file
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith("perturb_processed.h5ad"):
                    file_path = os.path.join(root, file)
                    if not self.silent:
                        print(f"Loading: {file_path}")
                    return anndata.read_h5ad(file_path)
        
        raise FileNotFoundError(f"Could not find 'perturb_processed.h5ad' in {self.path}")

    def __repr__(self):
        """Return a string representation of the object."""
        return (f"PertDataset object\n"
                f"    name: {self.name}\n"
                f"    cache_dir_path: {self.cache_dir_path}\n"
                f"    path: {self.path}\n"
                f"    adata: {self.adata}")
