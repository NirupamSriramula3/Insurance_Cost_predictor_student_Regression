import sys
import os

# ============================================================
# Add src to Python path
# ============================================================

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "src"
    )
)


# ============================================================
# Import Project Modules
# ============================================================

from insurance_cost_predicter.logger import logger

from insurance_cost_predicter.data_ingestion import (
    load_data,
    clean_data
)

from insurance_cost_predicter.data_eda import (
    run_all_eda
)

from insurance_cost_predicter.model_trainer import (
    prepare_features,
    train_and_evaluate,
    save_model
)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():

    logger.info("==========================================")
    logger.info("Regression Pipeline Started")
    logger.info("==========================================")


    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    data_path = (
        "artifacts/data/raw/"
        "health_insurance_cost_raw.csv"
    )

    df = load_data(data_path)


    # ========================================================
    # 2. CLEAN DATA
    #    Duplicates & Missing Values
    # ========================================================

    df = clean_data(df)


    # ========================================================
    # 3. TARGET DEFINITION
    # ========================================================

    target = "Annual_Insurance_Cost"


    # ========================================================
    # 4. RUN COMPLETE EDA
    #
    # Generates:
    #   - Numerical Histograms
    #   - Categorical Pie/Donut Charts
    #   - Numerical vs Numerical Scatter Plots
    #   - Categorical vs Numerical Box Plots
    # ========================================================

    run_all_eda(
        df,
        target_col=target
    )


    # ========================================================
    # 5. FEATURE ENGINEERING & SPLIT
    # ========================================================

    X, y = prepare_features(
        df,
        target_col=target,
        drop_cols=[
            "Customer_ID","Customer_Reference_Code"
        ]
    )


    # ========================================================
    # 6. MODEL TRAINING & EVALUATION
    #
    # Random Forest Regressor
    # ========================================================

    model, feature_names = train_and_evaluate(
        X,
        y
    )


    # ========================================================
    # 7. SAVE MODEL ARTIFACT
    # ========================================================

    save_model(
        model,
        feature_names,
        "artifacts/models/regression_rf_model.joblib"
    )


    # ========================================================
    # PIPELINE COMPLETED
    # ========================================================

    logger.info(
        "Regression Pipeline completed successfully"
    )

    print(
        "\nTraining and artifact generation complete."
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline()