import pathlib
import urllib.request
import tarfile
from argparse import ArgumentParser
import logging


def download_file(url, destination='.'):
    """Download a file from a URL to a specified destination."""
    try:
        # Download the file from the URL
        logging.info(f"Downloading file from: {url}")

        urllib.request.urlretrieve(url, destination)
        logging.info(f"Download successful!")
    except Exception as e:
        logging.error(f"Error downloading file: {e}")


def extract_tar_gz(file_path, extract_path='.'):
    """Extract a .tar.gz file to the specified directory."""
    try:
        with tarfile.open(file_path, 'r') as tar:
            tar.extractall(path=extract_path)  #==== Extract all contents to the specified directory
        logging.info(f"Extracted files to: {extract_path}")
    except tarfile.TarError as e:
        logging.error(f"Error extracting file: {e}")


def download_and_extract(data_name: str) -> None:
    data_name2url = {
        'my_mnist': 'https://github.com/leojlhsieh/SZZDist_deploy/releases/download/v1/my_mnist.tar.gz',
        'my_fashion_mnist': 'https://github.com/leojlhsieh/SZZDist_deploy/releases/download/v1/my_fashion_mnist.tar.gz',
        'my_cifar10': 'https://github.com/leojlhsieh/SZZDist_deploy/releases/download/v1/my_cifar10.tar.gz',
        'my_imagenette': 'https://github.com/leojlhsieh/SZZDist_deploy/releases/download/v1/my_imagenette.tar.gz',
    }
    assert data_name in data_name2url, f"{data_name} not supprt. choices=['my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette']"
    url = data_name2url[data_name]

    # If the folder 'data' is not in the parent parent of this script, create it
    data_dir = pathlib.Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    # If ' my_mnist' folder is not in data_dir, download it from https://github.com/leojlhsieh/SZZDist_deploy/releases/download/v1/my_mnist.tar.gz
    dataset_dir = data_dir / data_name
    dataset_file_dir = data_dir / f'{data_name}.tar.gz'
    if not dataset_dir.exists():
        if not dataset_file_dir.exists():
            download_file(url, dataset_file_dir)
        extract_tar_gz(dataset_file_dir, data_dir)
        dataset_file_dir.unlink()  # Delete the downloaded .tar.gz file

    return dataset_dir


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_name", type=str, choices=['my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette'])  # , required=True)
    args = parser.parse_args()

    logging.info(f"Downloading and extracting {args.data_name} dataset")
    import time
    start_time = time.time()
    download_and_extract(args.data_name)
    logging.info(f"--- {time.time() - start_time} seconds ---")
