
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

from insurance_cost_predicter.logger import logger


# ============================================================
# 1. PREPARE FEATURES
# ============================================================

def prepare_features(
    df,
    target_col="Annual_Insurance_Cost",
    drop_cols=None
):

    logger.info("Preparing features and target")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    if drop_cols is None:
        drop_cols = []

    cols_to_drop = [target_col] + drop_cols

    cols_to_drop = [
        col for col in cols_to_drop
        if col in df.columns
    ]

    X = df.drop(columns=cols_to_drop)
    y = df[target_col]

    logger.info("Encoding categorical features")

    X = pd.get_dummies(X, drop_first=True)

    logger.info(f"Feature shape: {X.shape}")

    return X, y


# ============================================================
# 2. DEFINE MODELS
# ============================================================

def get_models():

    models = {
        "Linear Regression": LinearRegression(),

        "KNN Regressor": KNeighborsRegressor(
            n_neighbors=3,
            p=1,
            weights="distance"
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        )
    }

    # Include XGBoost when the package is installed.
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror"
        )

    except ImportError:
        logger.warning(
            "XGBoost is not installed. Skipping XGBoost."
        )

    return models


# ============================================================
# 3. TRAIN AND EVALUATE ALL MODELS
# ============================================================

def train_and_evaluate(X, y):

    logger.info("Splitting dataset")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    models = get_models()

    results = []
    trained_models = {}

    for name, model in models.items():

        logger.info(f"Training {name}")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(
            y_test, y_pred
        )

        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape
        })

        trained_models[name] = model

        logger.info(
            f"{name} evaluation completed. MAPE={mape:.4f}"
        )

    # ========================================================
    # Compare models by lowest MAPE
    # ========================================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="MAPE",
        ascending=True
    ).reset_index(drop=True)

    print("\n========== MODEL COMPARISON ==========\n")
    print(results_df.to_string(index=False, float_format="%.4f"))

    # ========================================================
    # Select best model
    # ========================================================

    best_model_name = results_df.loc[0, "Model"]
    best_model = trained_models[best_model_name]

    print("\n========== BEST MODEL ==========")
    print("Model:", best_model_name)
    print(f"MAPE: {results_df.loc[0, 'MAPE']:.4f}")
    print(
        f"MAPE percentage: "
        f"{results_df.loc[0, 'MAPE'] * 100:.2f}%"
    )

    logger.info(f"Best model selected: {best_model_name}")

    return best_model, X_train.columns.tolist(), results_df


# ============================================================
# 4. SAVE ONLY THE BEST MODEL
# ============================================================

def save_model(
    model,
    feature_names,
    model_name,
    path="artifacts/models/Annual_Insurance_Cost_model.joblib"
):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    artifact = {
        "model": model,
        "features": feature_names,
        "model_name": model_name
    }

    logger.info(f"Saving best model to {path}")

    joblib.dump(artifact, path, compress=3)

    print(f"\nBest model saved at: {path}")

    logger.info("Model saved successfully")


# ============================================================
# 5. COMPLETE TRAINING PIPELINE
# ============================================================

def run_training(df):

    X, y = prepare_features(
        df,
        target_col="Annual_Insurance_Cost",
        drop_cols=[
            "Customer_ID",
            "Customer_Reference_Code"
        ]
    )

    best_model, feature_names, results_df = train_and_evaluate(
        X, y
    )

    best_model_name = results_df.loc[0, "Model"]

    save_model(
        model=best_model,
        feature_names=feature_names,
        model_name=best_model_name
    )

    return results_df
