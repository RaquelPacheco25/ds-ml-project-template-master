"""
Script para dividir los datos en conjunto de entrenamiento y conjunto de prueba.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit


def split_and_save_data(raw_data_path: str, interim_data_path: str):
    # Crear carpeta de salida si no existe
    Path(interim_data_path).mkdir(parents=True, exist_ok=True)

    # Leer datos
    housing = pd.read_csv(raw_data_path)

    # Crear variable auxiliar para estratificación
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, float("inf")],
        labels=[1, 2, 3, 4, 5]
    )

    # Split estratificado
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        train_set = housing.loc[train_index].drop("income_cat", axis=1)
        test_set = housing.loc[test_index].drop("income_cat", axis=1)

    # Guardar archivos
    train_set.to_csv(Path(interim_data_path) / "train_set.csv", index=False)
    test_set.to_csv(Path(interim_data_path) / "test_set.csv", index=False)

    print("Datos divididos y guardados correctamente.")
    print(f"Train shape: {train_set.shape}")
    print(f"Test shape: {test_set.shape}")


if __name__ == "__main__":
    RAW_PATH = "data/raw/housing/housing.csv"
    INTERIM_PATH = "data/interim/"
    split_and_save_data(RAW_PATH, INTERIM_PATH)