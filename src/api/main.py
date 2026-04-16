from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="California Housing Price Prediction API",
    description="API básica para predecir precios de vivienda con XGBoost",
    version="1.0.0"
)


# =========================================================
# INPUT SCHEMA
# =========================================================

class HousingFeatures(BaseModel):
    longitude: float = Field(..., example=-122.23)
    latitude: float = Field(..., example=37.88)
    housing_median_age: float = Field(..., example=41.0)
    total_rooms: float = Field(..., example=880.0)
    total_bedrooms: float = Field(..., example=129.0)
    population: float = Field(..., example=322.0)
    households: float = Field(..., example=126.0)
    median_income: float = Field(..., example=8.3252)
    ocean_proximity: str = Field(..., example="NEAR BAY")


# =========================================================
# CARGA DE ARTIFACT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = PROJECT_ROOT / "models" / "xgb_final_artifact.joblib"

artifact = joblib.load(ARTIFACT_PATH)

model = artifact["model"]
imputer = artifact["imputer"]
scaler = artifact["scaler"]
numeric_original_cols = artifact["numeric_original_cols"]
numeric_engineered_cols = artifact["numeric_engineered_cols"]
selected_features = artifact["selected_features"]
ocean_categories = artifact["ocean_categories"]
cat_col = artifact["cat_col"]


# =========================================================
# UTILIDADES
# =========================================================

def clean_feature_names(columns):
    cleaned = []
    for col in columns:
        col = str(col)
        col = re.sub(r"[<>\[\]]", "", col)
        col = col.replace(" ", "_")
        col = col.replace(",", "_")
        cleaned.append(col)
    return cleaned


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    df["rooms_per_household"] = df["total_rooms"] / (df["households"] + eps)
    df["bedrooms_per_room"] = df["total_bedrooms"] / (df["total_rooms"] + eps)
    df["population_per_household"] = df["population"] / (df["households"] + eps)
    df["bedrooms_per_household"] = df["total_bedrooms"] / (df["households"] + eps)
    df["rooms_per_person"] = df["total_rooms"] / (df["population"] + eps)
    df["geo_interaction"] = df["latitude"] * df["longitude"]

    df["log_total_rooms"] = np.log1p(df["total_rooms"])
    df["log_total_bedrooms"] = np.log1p(df["total_bedrooms"])
    df["log_population"] = np.log1p(df["population"])
    df["log_households"] = np.log1p(df["households"])
    df["log_median_income"] = np.log1p(df["median_income"])

    return df


def one_hot_ocean(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    df = df.copy()

    for cat in categories:
        col_name = f"ocean_proximity_{cat}"
        df[col_name] = (df[cat_col] == cat).astype(int)

    df = df.drop(columns=[cat_col])
    df.columns = clean_feature_names(df.columns)
    return df


def preprocess_input(payload: HousingFeatures) -> pd.DataFrame:
    raw_df = pd.DataFrame([payload.model_dump()])

    # 1. imputación
    X_num = pd.DataFrame(
        imputer.transform(raw_df[numeric_original_cols]),
        columns=numeric_original_cols,
        index=raw_df.index,
    )

    X_clean = pd.concat([X_num, raw_df[[cat_col]].copy()], axis=1)

    # 2. feature engineering
    X_fe = add_features(X_clean)

    # 3. scaling numérico
    X_num_fe = X_fe[numeric_engineered_cols].copy()
    X_num_scaled = pd.DataFrame(
        scaler.transform(X_num_fe),
        columns=numeric_engineered_cols,
        index=X_fe.index,
    )

    # 4. one-hot categórica
    X_cat = X_fe[[cat_col]].copy()
    X_cat_ohe = one_hot_ocean(X_cat, ocean_categories)

    # 5. combinar
    X_final = pd.concat([X_num_scaled, X_cat_ohe], axis=1)

    # 6. asegurar columnas seleccionadas
    for col in selected_features:
        if col not in X_final.columns:
            X_final[col] = 0

    X_final = X_final[selected_features]

    return X_final


# =========================================================
# ENDPOINTS
# =========================================================

@app.get("/")
def health():
    return {
        "status": "ok",
        "message": "Housing prediction API is running"
    }


@app.post("/predict")
def predict(features: HousingFeatures):
    X_ready = preprocess_input(features)
    pred = model.predict(X_ready)[0]

    return {
        "predicted_median_house_value": round(float(pred), 2)
    }