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
dir = os.getcwd().parent
file.extractall(dir)