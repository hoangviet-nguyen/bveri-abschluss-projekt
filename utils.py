from pathlib import Path


def download_from_gdrive_and_extract_zip(file_id: str, save_path: Path, extract_path: Path):
    """
    Downloads a ZIP file from Google Drive using its file ID and extracts its contents to a specified directory.

    Args:
        file_id (str): The Google Drive file ID of the ZIP file to download.
        save_path (Path): The path where the downloaded ZIP file will be saved.
        extract_path (Path): The directory where the ZIP file will be extracted.
    """
    import os
    import zipfile

    import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    if not save_path.exists():
        gdown.download(url, str(save_path), quiet=False)
        print(f"File downloaded and saved to {save_path}")

    if not extract_path.exists():
        print(f"Starting to extract... {extract_path}")
        # Unzip the file
        with zipfile.ZipFile(save_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        print(f"File extracted to {extract_path}")