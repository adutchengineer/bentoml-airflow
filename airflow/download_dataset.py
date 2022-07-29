"""
This script will download the necessary data for the training to run.
Note: The download will automatically happen in train.py if it does not exist
"""

from io import BytesIO
import requests
import os
import zipfile

movielens_data_file_url = (
'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
)
req = requests.get(movielens_data_file_url)
print('Downloading Completed')

file = zipfile.ZipFile(BytesIO(req.content))
dir = os.getcwd()
file.extractall(dir)