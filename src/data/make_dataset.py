"""
Script para descargar y extraer los datos originales del proyecto.
"""

import urllib.request
import tarfile
from pathlib import Path


def fetch_housing_data(housing_url: str, housing_path: str):
    # Crear carpeta si no existe
    Path(housing_path).mkdir(parents=True, exist_ok=True)

    # Ruta del archivo descargado
    tgz_path = Path(housing_path) / "housing.tgz"

    print("Descargando dataset...")
    urllib.request.urlretrieve(housing_url, tgz_path)

    print("Extrayendo dataset...")
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)

    print("Dataset listo en:", housing_path)


if __name__ == "__main__":
    URL = "https://github.com/ageron/data/raw/main/housing.tgz"
    PATH = "data/raw/"
    fetch_housing_data(URL, PATH)