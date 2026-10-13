import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error

from insurance_cost_predicter.logger import logger


# ============================================================
# 1. PREPARE FEATURES
# ============================================================

def prepare_features(df, target_col, drop_cols=None):

    logger.info("Splitting features and target")

    if drop_cols is None:
        drop_cols = []

    # Drop target and unnecessary ID columns
    cols_to_drop = [target_col] + drop_cols

    cols_to_drop = [
        c for c in cols_to_drop
        if c in df.columns
    ]

    X = df.drop(columns=cols_to_drop)
    y = df[target_col]

    # ========================================================
    # Encode categorical variables
    # ========================================================

    logger.info("Encoding categorical variables")

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    return X, y


# ============================================================
# 2. TRAIN AND EVALUATE
# ============================================================

def train_and_evaluate(X, y):

    logger.info("Train-test split")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # ========================================================
    # Train Random Forest Regressor
    # ========================================================

    logger.info("Training Random Forest Regressor")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    # ========================================================
    # Predictions
    # ========================================================

    logger.info("Making predictions")

    y_pred = model.predict(X_test)

    # ========================================================
    # Regression Evaluation
    # ========================================================

    logger.info("Evaluating regression model")

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        y_pred
    )

    mape = mean_absolute_percentage_error(
        y_test,
        y_pred
    )

    # ========================================================
    # Display Results
    # ========================================================

    print("\n==============================")
    print("REGRESSION MODEL REPORT")
    print("==============================\n")

    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return model, X_train.columns.tolist()


# ============================================================
# 3. SAVE MODEL
# ============================================================

def save_model(
    model,
    feature_names,
    path="artifacts/models/regression_rf_model.joblib"
):

    logger.info(
        f"Saving model artifact to: {path}"
    )

    # Saving both model and feature column order
    # for prediction consistency
    artifact = {
        "model": model,
        "features": feature_names
    }

    joblib.dump(
        artifact,
        path
    )

    print(
        f"\nModel saved successfully at: {path}"
    )