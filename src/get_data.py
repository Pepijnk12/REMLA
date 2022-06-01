"""
Download data from a certain source
"""

import urllib.request
import zipfile

URL = 'https://drive.google.com/u/0/uc?id=1uIPEUJ78imo62rE8FjP45FbrYwAUcWQY&export=download'
EXTRACT_DIR = "data/external"

zip_path, _ = urllib.request.urlretrieve(URL)
with zipfile.ZipFile(zip_path, "r") as f:
    f.extractall(EXTRACT_DIR)
