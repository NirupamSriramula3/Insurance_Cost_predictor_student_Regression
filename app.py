import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Batch Insurance Cost Prediction",
    layout="wide"
)

st.title("💰 Batch Insurance Cost Prediction")

st.write(
    "Upload a CSV file containing customer records "
    "to predict annual insurance costs in bulk."
)


# ============================================================
# 1. LOAD MODEL ARTIFACT
# ============================================================

artifact = joblib.load(
    "artifacts/models/regression_rf_model.joblib"
)

model = artifact["model"]
model_features = artifact["features"]


# ============================================================
# 2. FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)


if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    st.subheader(
        f"Uploaded Data Preview "
        f"({test_df.shape[0]} rows, {test_df.shape[1]} columns)"
    )

    st.dataframe(
        test_df.head(),
        use_container_width=True
    )


    # ========================================================
    # 3. RUN BATCH PREDICTION
    # ========================================================

    if st.button(
        " Run Batch Prediction",
        type="primary"
    ):

        # ----------------------------------------------------
        # Exclude Target, ID, or Previous Prediction Columns
        # ----------------------------------------------------

        ignore_cols = [
            "Annual_Insurance_Cost",
            "Customer_ID",
            "Predicted_Annual_Insurance_Cost"
        ]

        test_df = test_df.drop(columns=['Customer_Reference_Code'])
        features_df = test_df.drop(
            columns=[
                c for c in ignore_cols
                if c in test_df.columns
            ]
        )


        # ----------------------------------------------------
        # Handle Missing Values
        # ----------------------------------------------------

        for col in features_df.columns:

            if pd.api.types.is_numeric_dtype(
                features_df[col]
            ):

                features_df[col] = features_df[col].fillna(
                    features_df[col].median()
                )

            else:

                mode_val = features_df[col].mode()

                features_df[col] = features_df[col].fillna(
                    mode_val[0]
                    if not mode_val.empty
                    else "Missing"
                )


        # ----------------------------------------------------
        # Encode Categorical Variables
        # ----------------------------------------------------

        encoded_df = pd.get_dummies(
            features_df
        )


        # ----------------------------------------------------
        # Align Features With Training Data
        # ----------------------------------------------------

        encoded_df = encoded_df.reindex(
            columns=model_features,
            fill_value=0
        )


        # ----------------------------------------------------
        # Make Regression Predictions
        # ----------------------------------------------------

        predictions = model.predict(
            encoded_df
        )


        # ----------------------------------------------------
        # Attach Predictions To Original Data
        # ----------------------------------------------------

        result_df = test_df.copy()

        result_df[
            "Predicted_Annual_Insurance_Cost"
        ] = predictions.round(2)


        st.success(
            "Batch Prediction Complete!"
        )


        # ====================================================
        # 4. SUMMARY KPIs
        # ====================================================

        total = len(result_df)

        average_prediction = (
            result_df[
                "Predicted_Annual_Insurance_Cost"
            ].mean()
        )

        minimum_prediction = (
            result_df[
                "Predicted_Annual_Insurance_Cost"
            ].min()
        )

        maximum_prediction = (
            result_df[
                "Predicted_Annual_Insurance_Cost"
            ].max()
        )


        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Customers",
            total
        )

        col2.metric(
            "Average Predicted Cost",
            f"{average_prediction:,.2f}"
        )

        col3.metric(
            "Minimum Predicted Cost",
            f"{minimum_prediction:,.2f}"
        )

        col4.metric(
            "Maximum Predicted Cost",
            f"{maximum_prediction:,.2f}"
        )


        # ====================================================
        # 5. SCORED TABLE PREVIEW
        # ====================================================

        st.subheader(
            "Predicted Results Preview"
        )

        cols_to_show = [
            "Predicted_Annual_Insurance_Cost"
        ] + [
            c for c in result_df.columns
            if c != "Predicted_Annual_Insurance_Cost"
        ]

        st.dataframe(
            result_df[cols_to_show],
            use_container_width=True
        )


        # ====================================================
        # 6. DOWNLOAD RESULTS
        # ====================================================

        csv_data = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Predicted Results CSV",
            data=csv_data,
            file_name="insurance_cost_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )