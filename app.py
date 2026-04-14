import json
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
# LangChain is optional at runtime.
try:
    from langchain_core.prompts import PromptTemplate
except Exception:
    PromptTemplate = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Optional Gemini-backed LLM path.
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None


# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Mental Health Early Warning Dashboard",
    page_icon="🧠",
    layout="wide",  # Strict requirement: wide layout
)

st.title("🧠 AI-Based Mental Health Early Warning & Smart Intervention System")
st.caption("Interactive ML pipeline dashboard using Streamlit + Scikit-Learn + LangChain")

# -----------------------------
# Session State Initialization
# -----------------------------
def init_session_state() -> None:
    """Initialize all state keys once to keep app logic predictable."""
    defaults = {
        "raw_df": None,
        "working_df": None,
        "feature_columns": [],
        "target_column": "risk_level",
        "selected_features": [],
        "pipeline": None,
        "X_test": None,
        "y_test": None,
        "y_pred": None,
        "label_mapping": None,
        "metrics": None,
        "cleaning_log": [],
        "cleaning_report": None,
        "removed_rows_df": None,
        "outlier_bounds_df": None,
        "pre_clean_df": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# -----------------------------
# Data Utility Functions
# -----------------------------
def detect_numeric_and_categorical(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """Return numeric and categorical feature columns excluding target."""
    feature_df = df.drop(columns=[target_col], errors="ignore")
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def handle_missing_values(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Impute numeric with median and categorical with mode, excluding target."""
    updated = df.copy()
    feature_cols = [c for c in updated.columns if c != target_col]
    if not feature_cols:
        return updated

    numeric_cols = updated[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        updated[numeric_cols] = num_imputer.fit_transform(updated[numeric_cols])

    for col in categorical_cols:
        if updated[col].isna().any():
            mode_series = updated[col].mode(dropna=True)
            fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            updated[col] = updated[col].fillna(fill_value)

    return updated


def remove_outliers_iqr(df: pd.DataFrame, target_col: str, multiplier: float = 1.5) -> pd.DataFrame:
    """Remove row outliers via IQR on numeric feature columns only."""
    updated = df.copy()
    numeric_cols = updated.drop(columns=[target_col], errors="ignore").select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        return updated

    mask = pd.Series(True, index=updated.index)
    for col in numeric_cols:
        q1 = updated[col].quantile(0.25)
        q3 = updated[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        mask &= updated[col].between(lower, upper, inclusive="both")

    return updated[mask].reset_index(drop=True)


def compute_iqr_outlier_mask(
    df: pd.DataFrame, target_col: str, multiplier: float = 1.5
) -> Tuple[pd.Series, pd.DataFrame]:
    """Return outlier keep-mask and bounds summary for each numeric feature."""
    numeric_cols = df.drop(columns=[target_col], errors="ignore").select_dtypes(include=[np.number]).columns
    mask = pd.Series(True, index=df.index)
    bounds_records = []

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            lower, upper = q1, q3
            col_mask = pd.Series(True, index=df.index)
        else:
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            col_mask = df[col].between(lower, upper, inclusive="both")

        mask &= col_mask
        bounds_records.append(
            {
                "feature": col,
                "q1": round(float(q1), 4),
                "q3": round(float(q3), 4),
                "iqr": round(float(iqr), 4),
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
                "flagged_rows_in_feature": int((~col_mask).sum()),
            }
        )

    bounds_df = pd.DataFrame(bounds_records)
    return mask, bounds_df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline for mixed-type tabular data."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def get_model(model_name: str):
    """Factory for supported model choices."""
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1200, random_state=42)
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=250, random_state=42)
    raise ValueError(f"Unsupported model selected: {model_name}")


def safe_json(obj: Dict) -> str:
    """Render a compact JSON string for prompt readability."""
    return json.dumps(obj, indent=2, default=str)


def compute_weighted_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute core weighted metrics for classification evaluation."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


# -----------------------------
# Sidebar - File Upload
# -----------------------------
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Using uploaded dataset.")

        if df.empty:
            st.error("Uploaded CSV is empty. Please upload a valid dataset.")
            st.stop()

        st.session_state.raw_df = df.copy()

        if st.session_state.working_df is None:
            st.session_state.working_df = df.copy()

        cols = st.session_state.working_df.columns.tolist()
        if st.session_state.target_column not in cols:
            if "risk_level" in cols:
                st.session_state.target_column = "risk_level"
            elif "treatment" in cols:
                st.session_state.target_column = "treatment"
            else:
                st.session_state.target_column = cols[-1]

        selected_target = st.sidebar.selectbox(
            "Select target column",
            options=cols,
            index=cols.index(st.session_state.target_column),
        )
        st.session_state.target_column = selected_target

        st.success("CSV loaded successfully.")

    except Exception as ex:
        st.error(f"Failed to read CSV: {ex}")
        st.stop()
else:
    st.info("Please upload a CSV file from the sidebar to start the ML pipeline.")
    st.stop()


# -----------------------------
# Main App Tabs (exactly 6)
# -----------------------------
tab_names = [
    "Data & EDA",
    "Cleaning & Engineering",
    "Feature Selection",
    "Model Training",
    "Performance",
    "Smart Intervention",
]

(tab1, tab2, tab3, tab4, tab5, tab6) = st.tabs(tab_names)


# -----------------------------
# Tab 1: Data & EDA
# -----------------------------
with tab1:
    st.subheader("Pipeline Steps Coverage")
    st.markdown(
        "\n".join(
            [
                "1. Input Data",
                "2. Exploratory Data Analysis (EDA)",
                "3. Data Engineering & Cleaning",
                "4. Feature Selection",
                "5. Data Split",
                "6. Model Selection",
                "7. Model Training",
                "8. K-Fold Validation",
                "9. Performance Metrics",
            ]
        )
    )

    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.working_df.head(20), use_container_width=True)

    st.subheader("Dataset Summary (describe)")
    st.dataframe(st.session_state.working_df.describe(include="all").transpose(), use_container_width=True)

    st.subheader("Correlation Heatmap")
    numeric_df = st.session_state.working_df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = numeric_df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)
    else:
        st.warning("Need at least 2 numeric columns to compute a correlation heatmap.")

    st.subheader("Custom Graph Builder")
    chart_type = st.selectbox(
        "Graph type",
        ["Scatter Plot", "Line Plot", "Bar Plot", "Histogram", "Box Plot"],
        key="eda_chart_type",
    )
    cols = st.session_state.working_df.columns.tolist()
    numeric_cols = st.session_state.working_df.select_dtypes(include=[np.number]).columns.tolist()

    x_col = st.selectbox("X-axis", cols, key="eda_x_col")
    y_candidates = [None] + cols
    y_col = st.selectbox("Y-axis (optional)", y_candidates, key="eda_y_col")
    hue_candidates = [None] + [c for c in cols if c != x_col]
    hue_col = st.selectbox("Color group (optional)", hue_candidates, key="eda_hue_col")

    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        if chart_type == "Scatter Plot":
            if y_col is None:
                st.info("Scatter plot needs both X and Y columns.")
            else:
                sns.scatterplot(
                    data=st.session_state.working_df,
                    x=x_col,
                    y=y_col,
                    hue=hue_col,
                    ax=ax,
                )
                st.pyplot(fig)
        elif chart_type == "Line Plot":
            if y_col is None:
                st.info("Line plot needs both X and Y columns.")
            else:
                sns.lineplot(
                    data=st.session_state.working_df,
                    x=x_col,
                    y=y_col,
                    hue=hue_col,
                    ax=ax,
                )
                st.pyplot(fig)
        elif chart_type == "Bar Plot":
            if y_col is None:
                sns.countplot(data=st.session_state.working_df, x=x_col, hue=hue_col, ax=ax)
            else:
                if y_col not in numeric_cols:
                    st.info("For Bar Plot with Y-axis, choose a numeric Y column.")
                else:
                    grouped = (
                        st.session_state.working_df.groupby(x_col, dropna=False)[y_col]
                        .mean()
                        .reset_index()
                    )
                    sns.barplot(data=grouped, x=x_col, y=y_col, ax=ax)
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)
        elif chart_type == "Histogram":
            if x_col not in numeric_cols:
                st.info("Histogram needs a numeric X column.")
            else:
                sns.histplot(
                    data=st.session_state.working_df,
                    x=x_col,
                    hue=hue_col,
                    kde=True,
                    bins=30,
                    ax=ax,
                )
                st.pyplot(fig)
        elif chart_type == "Box Plot":
            if x_col in numeric_cols and y_col is not None and y_col not in numeric_cols:
                sns.boxplot(data=st.session_state.working_df, x=y_col, y=x_col, ax=ax)
                st.pyplot(fig)
            elif y_col is not None and y_col in numeric_cols:
                sns.boxplot(data=st.session_state.working_df, x=x_col, y=y_col, ax=ax)
                ax.tick_params(axis="x", rotation=45)
                st.pyplot(fig)
            elif x_col in numeric_cols:
                sns.boxplot(data=st.session_state.working_df, y=x_col, ax=ax)
                st.pyplot(fig)
            else:
                st.info("Box plot needs at least one numeric column.")
    except Exception as ex:
        st.error(f"Could not render selected graph: {ex}")


# -----------------------------
# Tab 2: Cleaning & Engineering
# -----------------------------
with tab2:
    st.subheader("Automated Data Cleaning")
    target_col = st.session_state.target_column

    col_a, col_b = st.columns(2)
    with col_a:
        do_impute = st.checkbox("Handle Missing Values (Median/Mode Imputation)")
    with col_b:
        do_outlier = st.checkbox("Remove Outliers (IQR Method)")

    if st.button("Apply Cleaning Steps", type="primary"):
        try:
            updated_df = st.session_state.working_df.copy()
            st.session_state.pre_clean_df = updated_df.copy()
            original_shape = updated_df.shape
            logs = []
            report = {
                "rows_before": int(original_shape[0]),
                "rows_after": int(original_shape[0]),
                "columns_before": int(original_shape[1]),
                "columns_after": int(original_shape[1]),
                "missing_before_total": int(updated_df.isna().sum().sum()),
                "missing_after_total": int(updated_df.isna().sum().sum()),
                "rows_removed": 0,
            }
            st.session_state.removed_rows_df = None
            st.session_state.outlier_bounds_df = None

            if do_impute:
                before_na = int(updated_df.isna().sum().sum())
                updated_df = handle_missing_values(updated_df, target_col)
                after_na = int(updated_df.isna().sum().sum())
                logs.append(f"Missing values handled: {before_na} -> {after_na}")
                report["missing_before_total"] = before_na
                report["missing_after_total"] = after_na

            if do_outlier:
                before_rows = updated_df.shape[0]
                keep_mask, bounds_df = compute_iqr_outlier_mask(updated_df, target_col=target_col)
                removed_rows = updated_df.loc[~keep_mask].copy()
                st.session_state.removed_rows_df = removed_rows
                st.session_state.outlier_bounds_df = bounds_df
                updated_df = updated_df.loc[keep_mask].reset_index(drop=True)
                after_rows = updated_df.shape[0]
                logs.append(f"Outlier removal rows: {before_rows} -> {after_rows}")
                report["rows_removed"] = int(before_rows - after_rows)

            st.session_state.working_df = updated_df
            st.session_state.cleaning_log = logs
            report["rows_after"] = int(updated_df.shape[0])
            report["columns_after"] = int(updated_df.shape[1])
            report["missing_after_total"] = int(updated_df.isna().sum().sum())
            st.session_state.cleaning_report = report

            st.success("Cleaning/engineering pipeline applied.")
            st.write(f"Original shape: {original_shape}")
            st.write(f"Updated shape: {updated_df.shape}")
            if logs:
                st.info(" | ".join(logs))

        except Exception as ex:
            st.error(f"Error during cleaning: {ex}")

    st.write("Current dataset shape:", st.session_state.working_df.shape)

    if st.session_state.cleaning_report is not None:
        st.subheader("Cleaning Impact Summary")
        rep = st.session_state.cleaning_report
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows Before", rep["rows_before"])
        c2.metric("Rows After", rep["rows_after"])
        c3.metric("Rows Removed", rep["rows_removed"])
        c4.metric("Missing Values Filled", rep["missing_before_total"] - rep["missing_after_total"])

        if st.session_state.pre_clean_df is not None:
            before_missing = st.session_state.pre_clean_df.isna().sum().rename("before_missing")
            after_missing = st.session_state.working_df.isna().sum().rename("after_missing")
            miss_cmp = pd.concat([before_missing, after_missing], axis=1).reset_index()
            miss_cmp.columns = ["feature", "before_missing", "after_missing"]
            st.write("Missing value comparison by feature")
            st.dataframe(miss_cmp, use_container_width=True)

        if st.session_state.outlier_bounds_df is not None and not st.session_state.outlier_bounds_df.empty:
            st.write("IQR bounds used for outlier removal")
            st.dataframe(st.session_state.outlier_bounds_df, use_container_width=True)

        removed_df = st.session_state.removed_rows_df
        if removed_df is not None:
            st.write(f"Removed rows preview ({removed_df.shape[0]} rows removed)")
            st.dataframe(removed_df.head(50), use_container_width=True)

            num_cols = removed_df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                feature_for_removed = st.selectbox(
                    "See removed values for feature",
                    num_cols,
                    key="removed_values_feature",
                )
                st.dataframe(
                    removed_df[[feature_for_removed]].sort_values(by=feature_for_removed).head(100),
                    use_container_width=True,
                )

        if st.session_state.pre_clean_df is not None:
            st.subheader("Before vs After Distribution")
            before_df = st.session_state.pre_clean_df
            after_df = st.session_state.working_df
            num_cols = before_df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                dist_col = st.selectbox("Feature for comparison", num_cols, key="dist_compare_col")
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.kdeplot(before_df[dist_col], label="Before", fill=True, alpha=0.3, ax=ax)
                sns.kdeplot(after_df[dist_col], label="After", fill=True, alpha=0.3, ax=ax)
                ax.set_title(f"Distribution Shift: {dist_col}")
                ax.legend()
                st.pyplot(fig)


# -----------------------------
# Tab 3: Feature Selection
# -----------------------------
with tab3:
    st.subheader("Feature Selection")
    target_col = st.session_state.target_column

    if target_col not in st.session_state.working_df.columns:
        st.error(f"Target column '{target_col}' not found in dataset.")
    else:
        all_features = [c for c in st.session_state.working_df.columns if c != target_col]

        feature_mode = st.radio(
            "Selection mode",
            ["Keep selected features", "Drop selected features"],
            horizontal=True,
        )

        selected = st.multiselect(
            "Choose columns",
            options=all_features,
            default=all_features,
        )

        if feature_mode == "Keep selected features":
            chosen_features = selected
        else:
            chosen_features = [c for c in all_features if c not in selected]

        st.session_state.selected_features = chosen_features

        st.write(f"Total available features: {len(all_features)}")
        st.write(f"Selected features for modeling: {len(chosen_features)}")
        st.code(", ".join(chosen_features) if chosen_features else "No features selected.")


# -----------------------------
# Tab 4: Model Training
# -----------------------------
with tab4:
    st.subheader("Train ML Model")
    target_col = st.session_state.target_column

    model_choice = st.selectbox("Select model", ["Logistic Regression", "Random Forest"])
    test_size_pct = st.slider("Test set percentage", min_value=10, max_value=40, value=20, step=5)
    k_folds = st.slider("K-Fold splits", min_value=3, max_value=10, value=5, step=1)

    if st.button("Start Training Pipeline", type="primary"):
        try:
            df_train = st.session_state.working_df.copy()

            if target_col not in df_train.columns:
                st.error(f"Target column '{target_col}' missing. Cannot train model.")
                st.stop()

            chosen_features = st.session_state.selected_features
            if not chosen_features:
                st.error(
                    "No features selected. Please go to 'Feature Selection' tab and select at least one feature."
                )
                st.stop()

            for col in chosen_features:
                if col not in df_train.columns:
                    st.error(f"Selected feature '{col}' no longer exists in dataset.")
                    st.stop()

            X = df_train[chosen_features]
            y = df_train[target_col]

            if y.nunique() < 2:
                st.error("Target must contain at least 2 classes for classification.")
                st.stop()

            preprocessor = build_preprocessor(X)
            estimator = get_model(model_choice)

            model_pipeline = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", estimator),
                ]
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size_pct / 100.0,
                random_state=42,
                stratify=y if y.nunique() > 1 else None,
            )

            model_pipeline.fit(X_train, y_train)
            y_pred = model_pipeline.predict(X_test)

            # Persist training artifacts in session state.
            st.session_state.pipeline = model_pipeline
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.feature_columns = chosen_features

            holdout_metrics = compute_weighted_metrics(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            cv_strategy = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            cv_scores = cross_validate(
                model_pipeline,
                X,
                y,
                cv=cv_strategy,
                scoring=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
                n_jobs=None,
            )
            cv_summary = {
                "k_folds": int(k_folds),
                "accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
                "accuracy_std": float(np.std(cv_scores["test_accuracy"])),
                "precision_weighted_mean": float(np.mean(cv_scores["test_precision_weighted"])),
                "precision_weighted_std": float(np.std(cv_scores["test_precision_weighted"])),
                "recall_weighted_mean": float(np.mean(cv_scores["test_recall_weighted"])),
                "recall_weighted_std": float(np.std(cv_scores["test_recall_weighted"])),
                "f1_weighted_mean": float(np.mean(cv_scores["test_f1_weighted"])),
                "f1_weighted_std": float(np.std(cv_scores["test_f1_weighted"])),
                "raw_folds": {
                    "accuracy": cv_scores["test_accuracy"].tolist(),
                    "precision_weighted": cv_scores["test_precision_weighted"].tolist(),
                    "recall_weighted": cv_scores["test_recall_weighted"].tolist(),
                    "f1_weighted": cv_scores["test_f1_weighted"].tolist(),
                },
            }

            st.session_state.metrics = {
                "model": model_choice,
                "target": target_col,
                "feature_count": int(len(chosen_features)),
                "features": chosen_features,
                "train_rows": int(X_train.shape[0]),
                "test_rows": int(X_test.shape[0]),
                "test_size_pct": int(test_size_pct),
                "holdout": holdout_metrics,
                "classification_report": class_report,
                "kfold": cv_summary,
            }

            st.success("Training completed and saved in session.")
            st.write("Data split complete:", {
                "train_rows": int(X_train.shape[0]),
                "test_rows": int(X_test.shape[0]),
                "test_size_pct": int(test_size_pct),
            })
            st.write("Holdout metrics:", holdout_metrics)
            st.write(
                "K-Fold summary:",
                {
                    "k_folds": cv_summary["k_folds"],
                    "accuracy_mean": round(cv_summary["accuracy_mean"], 4),
                    "f1_weighted_mean": round(cv_summary["f1_weighted_mean"], 4),
                },
            )
            st.balloons()

        except Exception as ex:
            st.error(f"Training pipeline failed: {ex}")


# -----------------------------
# Tab 5: Performance
# -----------------------------
with tab5:
    st.subheader("Model Performance")

    if st.session_state.pipeline is None:
        st.warning("No trained model found. Please train a model in 'Model Training' tab.")
    else:
        metrics = st.session_state.metrics or {}
        holdout = metrics.get("holdout", {})
        kfold = metrics.get("kfold", {})

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model", str(metrics.get("model", "N/A")))
        m2.metric("Holdout Accuracy", f"{holdout.get('accuracy', 0):.4f}")
        m3.metric("Holdout F1 (Weighted)", f"{holdout.get('f1_weighted', 0):.4f}")
        m4.metric("K-Fold Accuracy (Mean)", f"{kfold.get('accuracy_mean', 0):.4f}")

        st.subheader("Holdout Set Metrics")
        holdout_table = pd.DataFrame(
            [
                {
                    "accuracy": holdout.get("accuracy", 0.0),
                    "precision_weighted": holdout.get("precision_weighted", 0.0),
                    "recall_weighted": holdout.get("recall_weighted", 0.0),
                    "f1_weighted": holdout.get("f1_weighted", 0.0),
                }
            ]
        )
        st.dataframe(holdout_table.style.format("{:.4f}"), use_container_width=True)

        if kfold:
            st.subheader("K-Fold Cross Validation Summary")
            kfold_table = pd.DataFrame(
                [
                    {
                        "k_folds": kfold.get("k_folds", 0),
                        "accuracy_mean": kfold.get("accuracy_mean", 0.0),
                        "accuracy_std": kfold.get("accuracy_std", 0.0),
                        "precision_weighted_mean": kfold.get("precision_weighted_mean", 0.0),
                        "precision_weighted_std": kfold.get("precision_weighted_std", 0.0),
                        "recall_weighted_mean": kfold.get("recall_weighted_mean", 0.0),
                        "recall_weighted_std": kfold.get("recall_weighted_std", 0.0),
                        "f1_weighted_mean": kfold.get("f1_weighted_mean", 0.0),
                        "f1_weighted_std": kfold.get("f1_weighted_std", 0.0),
                    }
                ]
            )
            st.dataframe(kfold_table.style.format("{:.4f}"), use_container_width=True)

            fold_scores = kfold.get("raw_folds", {})
            if fold_scores:
                st.write("Per-fold scores")
                st.dataframe(pd.DataFrame(fold_scores), use_container_width=True)

        report_dict = metrics.get("classification_report")
        if report_dict:
            st.subheader("Class-wise Classification Report (Holdout)")
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df, use_container_width=True)

        st.subheader("Confusion Matrix")
        try:
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            if y_test is None or y_pred is None:
                st.info("Confusion matrix is available after a fresh train run in this session.")
                st.stop()

            labels = sorted(pd.unique(pd.concat([pd.Series(y_test), pd.Series(y_pred)], axis=0)).tolist())
            cm = confusion_matrix(y_test, y_pred, labels=labels)

            fig, ax = plt.subplots(figsize=(8, 5))
            display_labels = [str(label) for label in labels]
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            disp.plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
        except Exception as ex:
            st.error(f"Could not render confusion matrix: {ex}")


# -----------------------------
# Tab 6: Smart Intervention
# -----------------------------
with tab6:
    st.subheader("GenAI Smart Intervention")

    if st.session_state.pipeline is None:
        st.warning("Train a model first to enable intervention predictions.")
    else:
        feature_cols = st.session_state.feature_columns
        if not feature_cols:
            st.error("No trained feature set found in session.")
            st.stop()

        current_df = st.session_state.working_df

        with st.form("intervention_form"):
            st.write("Enter hypothetical patient profile:")
            patient_input = {}

            # Build dynamic inputs based on feature types and observed ranges.
            for col in feature_cols:
                series = current_df[col]

                if pd.api.types.is_numeric_dtype(series):
                    col_min = float(np.nanmin(series.values)) if not series.isna().all() else 0.0
                    col_max = float(np.nanmax(series.values)) if not series.isna().all() else 100.0
                    default = float(np.nanmedian(series.values)) if not series.isna().all() else 0.0

                    # Use slider for bounded ranges; number_input fallback for huge/flat ranges.
                    if col_max > col_min and (col_max - col_min) <= 500:
                        patient_input[col] = st.slider(
                            label=col,
                            min_value=float(round(col_min, 2)),
                            max_value=float(round(col_max, 2)),
                            value=float(round(default, 2)),
                            step=0.1,
                        )
                    else:
                        patient_input[col] = st.number_input(
                            label=col,
                            value=float(round(default, 2)),
                            step=0.1,
                        )
                else:
                    # Categorical fields use selectbox from known unique values.
                    unique_vals = series.dropna().astype(str).unique().tolist()
                    if not unique_vals:
                        unique_vals = ["Unknown"]
                    patient_input[col] = st.selectbox(col, unique_vals)

            api_key = st.text_input(
                "Gemini API Key (required for Smart Intervention plan)",
                type="password",
                help="The intervention plan is generated only when an API key is provided.",
            )

            submitted = st.form_submit_button("Generate Risk & Smart Intervention")

        if submitted:
            try:
                # 1) Predict risk via trained ML model
                input_df = pd.DataFrame([patient_input])
                pred = st.session_state.pipeline.predict(input_df)[0]

                # Normalize output to Low/Medium/High if model uses numeric classes.
                if str(pred) not in {"Low", "Medium", "High"}:
                    mapped = {0: "Low", 1: "Medium", 2: "High"}
                    pred_label = mapped.get(pred, str(pred))
                else:
                    pred_label = str(pred)

                st.success(f"Predicted Mental Health Risk Level: **{pred_label}**")

                # 2) Build intervention prompt payload
                template_text = """
You are a clinical-support AI assistant for early mental health intervention.

Patient risk level: {risk_level}
Patient feature profile:
{feature_payload}

Generate a personalized, practical Smart Intervention Plan with:
1) Immediate actions for next 24 hours
2) 7-day routine suggestions
3) Escalation criteria (when to seek professional help urgently)
4) Short motivational message

Keep tone supportive, actionable, and safe. Avoid diagnosis claims.
""".strip()

                feature_payload = safe_json(patient_input)

                if not api_key:
                    st.info("Add a Gemini API key to generate the Smart Intervention plan.")
                    st.stop()

                if ChatGoogleGenerativeAI is None or PromptTemplate is None:
                    st.error(
                        "LangChain Gemini dependencies are unavailable. Install requirements to enable plan generation."
                    )
                    st.stop()

                prompt_template = PromptTemplate.from_template(template_text)
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=api_key,
                    temperature=0.3,
                )
                chain = prompt_template | llm
                response = chain.invoke(
                    {"risk_level": pred_label, "feature_payload": feature_payload}
                )
                plan_text = response.content if hasattr(response, "content") else str(response)

                st.markdown("#### Personalized Smart Intervention Plan")
                st.markdown(plan_text)

            except Exception as ex:
                st.error(f"Smart intervention generation failed: {ex}")