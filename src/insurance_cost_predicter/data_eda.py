import os
import math
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from insurance_cost_predicter.logger import logger

warnings.filterwarnings(action='ignore')

# Ensure visual artifacts directory exists
OUTPUT_DIR = "artifacts/visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. UNIVARIATE ANALYSIS - NUMERICAL (df.hist)
# ============================================================

def plot_univariate_numerical(
    df,
    save_filename='Univariate Analysis - Numerical Variables Distribution.png'
):
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    logger.info("Generating Univariate Numerical Distributions")

    df_num = df.select_dtypes(include=['int64', 'float64'])
    df_num = df_num[
        [col for col in df_num.columns if 'id' not in col.lower()]
    ]

    plt.figure(figsize=(15, 12))

    df_num.hist(
        figsize=(15, 12),
        bins=30,
        edgecolor='black',
        grid=False
    )

    plt.suptitle(
        'Univariate Numerical Distributions',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

    print(f"Saved: {save_path}")
    logger.info(f"Saved numerical univariate chart to {save_path}")


# ============================================================
# 2. UNIVARIATE ANALYSIS - CATEGORICAL (Donut Composition)
# ============================================================

def plot_univariate_categorical(
    df,
    save_filename='Univariate Analysis - Categorical Variables Composition.png'
):
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    logger.info("Generating Univariate Categorical Composition")

    df_cat = df.select_dtypes(include=['object', 'category'])
    df_cat = df_cat.drop(columns=['Customer_ID','Customer_Reference_Code'])
    cat_columns = [
        col for col in df_cat.columns
        if 'id' not in col.lower()
    ]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15, 12)
    )

    axes = axes.flatten()

    for i, col in enumerate(cat_columns[:9]):

        counts = df_cat[col].value_counts()

        axes[i].pie(
            counts.values,
            labels=counts.index,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.75,
            wedgeprops=dict(
                width=0.45,
                edgecolor='white'
            )
        )

        axes[i].set_title(
            col,
            fontsize=12,
            fontweight='bold'
        )

    for j in range(len(cat_columns[:9]), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(
        'Univariate Categorical Composition',
        fontsize=15,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

    print(f"Saved: {save_path}")
    logger.info(f"Saved categorical univariate chart to {save_path}")


# ============================================================
# 3. BIVARIATE ANALYSIS - NUMERICAL VS NUMERICAL
#    Scatter Plots
# ============================================================

def plot_bivariate_numerical(
    df,
    target_col='Target',
    save_filename='Bivariate Analysis - Numerical Variables vs Target Variable Scatter Plots.png'
):
    save_path = os.path.join(OUTPUT_DIR, save_filename)

    logger.info(
        f"Generating Bivariate Numerical Scatter Plots against {target_col}"
    )

    # Select numerical columns
    df_num = df.select_dtypes(
        include=['int64', 'float64']
    )

    # Remove ID columns and target column
    num_cols = [
        c for c in df_num.columns
        if 'id' not in c.lower()
        and c != target_col
    ]

    n_features = len(num_cols)

    if n_features == 0:
        logger.warning(
            "No numerical feature columns available for scatter plots."
        )
        return

    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(16, n_rows * 4)
    )

    # Make axes iterable when there is only one subplot
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(num_cols):

        ax = axes[i]

        sns.scatterplot(
            data=df,
            x=col,
            y=target_col,
            ax=ax,
            alpha=0.7
        )

        ax.set_title(
            f'{col} vs. {target_col}',
            fontsize=11,
            fontweight='bold'
        )

        ax.set_xlabel(
            col,
            fontsize=10
        )

        ax.set_ylabel(
            target_col,
            fontsize=10
        )

        ax.grid(
            linestyle='--',
            alpha=0.4
        )

    # Remove unused subplots
    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(
        f'Bivariate Analysis: Numerical Features vs. {target_col}',
        fontsize=15,
        fontweight='bold',
        y=1.01
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

    print(f"Saved: {save_path}")

    logger.info(
        f"Saved numerical bivariate scatter plots to {save_path}"
    )


# ============================================================
# 4. BIVARIATE ANALYSIS - CATEGORICAL VS NUMERICAL TARGET
#    Multi-Box Plots
# ============================================================

def plot_bivariate_categorical(
    df,
    target_col='Target',
    save_filename='Bivariate Analysis - Categorical Variables vs Target Variable Box Plots.png'
):
    save_path = os.path.join(OUTPUT_DIR, save_filename)

    logger.info(
        f"Generating Bivariate Box Plots against {target_col}"
    )

    # Select categorical columns
    df_cat = df.select_dtypes(
        include=['object', 'category']
    )
    df_cat = df_cat.drop(columns=['Customer_ID','Customer_Reference_Code'])
    # Remove ID columns
    cat_columns = [
        col for col in df_cat.columns
        if 'id' not in col.lower()
        and col != target_col
    ]

    n_features = len(cat_columns)

    if n_features == 0:
        logger.warning(
            "No categorical feature columns available for box plots."
        )
        return

    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(16, n_rows * 4)
    )

    # Make axes iterable when there is only one subplot
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(cat_columns):

        ax = axes[i]

        sns.boxplot(
            data=df,
            x=col,
            y=target_col,
            ax=ax,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": "6"
            }
        )

        ax.set_title(
            f'{target_col} by {col}',
            fontsize=11,
            fontweight='bold'
        )

        ax.set_xlabel(
            col,
            fontsize=10
        )

        ax.set_ylabel(
            target_col,
            fontsize=10
        )

        ax.tick_params(
            axis='x',
            rotation=20
        )

        ax.grid(
            axis='y',
            linestyle='--',
            alpha=0.5
        )

    # Remove unused subplots
    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(
        f'Bivariate Analysis: Categorical Features vs. {target_col}',
        fontsize=15,
        fontweight='bold',
        y=1.01
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

    print(f"Saved: {save_path}")

    logger.info(
        f"Saved categorical bivariate box plots to {save_path}"
    )


# ============================================================
# 5. MASTER EXECUTION FUNCTION
# ============================================================

def run_all_eda(
    df,
    target_col='Target'
):
    """
    Executes all 4 EDA steps for a regression problem
    and writes charts to artifacts/visuals/

    EDA Structure:

    1. Univariate Numerical
       -> Histograms

    2. Univariate Categorical
       -> Donut/Pie Charts

    3. Bivariate Numerical vs Numerical
       -> Scatter Plots

    4. Bivariate Categorical vs Numerical Target
       -> Box Plots
    """

    logger.info(
        "Starting Full Regression EDA Execution"
    )

    # 1. Univariate numerical
    plot_univariate_numerical(df)

    # 2. Univariate categorical
    plot_univariate_categorical(df)

    # 3. Numerical vs Numerical
    plot_bivariate_numerical(
        df,
        target_col=target_col
    )

    # 4. Categorical vs Numerical Target
    plot_bivariate_categorical(
        df,
        target_col=target_col
    )

    logger.info(
        "Full Regression EDA Execution Completed Successfully"
    )