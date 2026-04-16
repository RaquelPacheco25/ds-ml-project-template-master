from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables derivadas a partir de las variables originales.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con variables explicativas limpias.

    Returns
    -------
    pd.DataFrame
        DataFrame con nuevas variables derivadas.
    """
    df = df.copy()
    eps = 1e-6  # evita divisiones por cero

    # Variables pedidas en la consigna
    df["rooms_per_household"] = df["total_rooms"] / (df["households"] + eps)
    df["bedrooms_per_room"] = df["total_bedrooms"] / (df["total_rooms"] + eps)
    df["population_per_household"] = df["population"] / (df["households"] + eps)

    # Variables adicionales defendibles
    df["bedrooms_per_household"] = df["total_bedrooms"] / (df["households"] + eps)
    df["rooms_per_person"] = df["total_rooms"] / (df["population"] + eps)
    df["geo_interaction"] = df["latitude"] * df["longitude"]

    # Transformaciones logarítmicas
    df["log_total_rooms"] = np.log1p(df["total_rooms"])
    df["log_total_bedrooms"] = np.log1p(df["total_bedrooms"])
    df["log_population"] = np.log1p(df["population"])
    df["log_households"] = np.log1p(df["households"])
    df["log_median_income"] = np.log1p(df["median_income"])

    return df


def load_data(train_path: Path, test_path: Path):
    """
    Carga train y test desde disco.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def split_features_target(df: pd.DataFrame, target_col: str):
    """
    Separa variables explicativas y variable objetivo.
    """
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y


def build_preprocessor(num_attribs: list, cat_attribs: list) -> ColumnTransformer:
    """
    Construye el pipeline de preprocesamiento:
    - escalado para variables numéricas
    - one-hot encoding para variables categóricas
    """
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return preprocessor


def main():
    # =========================================================
    # RUTAS
    # =========================================================
    project_root = Path(__file__).resolve().parents[2]

    train_path = project_root / "data" / "interim" / "train_set.csv"
    test_path = project_root / "data" / "interim" / "test_set.csv"

    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # CARGA DE DATOS
    # =========================================================
    train, test = load_data(train_path, test_path)

    target_col = "median_house_value"

    X_train, y_train = split_features_target(train, target_col)
    X_test, y_test = split_features_target(test, target_col)

    print(f"X_train original: {X_train.shape}")
    print(f"X_test original: {X_test.shape}")

    # =========================================================
    # IMPUTACIÓN NUMÉRICA
    # =========================================================
    num_cols_original = X_train.drop(columns=["ocean_proximity"]).columns.tolist()
    cat_cols = ["ocean_proximity"]

    imputer = SimpleImputer(strategy="median")

    X_train_num = pd.DataFrame(
        imputer.fit_transform(X_train[num_cols_original]),
        columns=num_cols_original,
        index=X_train.index
    )

    X_test_num = pd.DataFrame(
        imputer.transform(X_test[num_cols_original]),
        columns=num_cols_original,
        index=X_test.index
    )

    X_train_clean = pd.concat([X_train_num, X_train[cat_cols]], axis=1)
    X_test_clean = pd.concat([X_test_num, X_test[cat_cols]], axis=1)

    # =========================================================
    # FEATURE ENGINEERING
    # =========================================================
    X_train_fe = add_features(X_train_clean)
    X_test_fe = add_features(X_test_clean)

    # =========================================================
    # PREPROCESAMIENTO FINAL
    # =========================================================
    num_attribs = X_train_fe.drop(columns=["ocean_proximity"]).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    preprocessor = build_preprocessor(num_attribs, cat_attribs)

    X_train_prepared = preprocessor.fit_transform(X_train_fe)
    X_test_prepared = preprocessor.transform(X_test_fe)

    # =========================================================
    # NOMBRES FINALES DE FEATURES
    # =========================================================
    ohe = preprocessor.named_transformers_["cat"]["onehot"]
    cat_feature_names = ohe.get_feature_names_out(cat_attribs)

    final_feature_names = num_attribs + list(cat_feature_names)

    X_train_prepared_df = pd.DataFrame(
        X_train_prepared,
        columns=final_feature_names,
        index=X_train_fe.index
    )

    X_test_prepared_df = pd.DataFrame(
        X_test_prepared,
        columns=final_feature_names,
        index=X_test_fe.index
    )

    # =========================================================
    # VALIDACIONES
    # =========================================================
    print(f"X_train procesado: {X_train_prepared_df.shape}")
    print(f"X_test procesado: {X_test_prepared_df.shape}")
    print(f"Nulos en X_train procesado: {X_train_prepared_df.isnull().sum().sum()}")
    print(f"Nulos en X_test procesado: {X_test_prepared_df.isnull().sum().sum()}")

    # =========================================================
    # GUARDADO DE ARCHIVOS
    # =========================================================
    X_train_prepared_df.to_csv(output_dir / "X_train_prepared.csv", index=False)
    X_test_prepared_df.to_csv(output_dir / "X_test_prepared.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print("\nArchivos guardados en:")
    print(output_dir / "X_train_prepared.csv")
    print(output_dir / "X_test_prepared.csv")
    print(output_dir / "y_train.csv")
    print(output_dir / "y_test.csv")


if __name__ == "__main__":
    main()
