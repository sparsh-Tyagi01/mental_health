import argparse
import numpy as np
import pandas as pd


def generate_synthetic_mental_health_data(n_rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic mental health dataset with realistic correlations."""
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 61, n_rows)
    work_hours_per_week = np.clip(rng.normal(45, 10, n_rows), 20, 80)
    sleep_hours = np.clip(rng.normal(7, 1.5, n_rows), 3, 10)
    stress_level = np.clip(rng.normal(6, 2.2, n_rows), 1, 10)
    physical_activity_hours = np.clip(rng.normal(3, 1.8, n_rows), 0, 10)
    social_interaction_score = np.clip(rng.normal(5.5, 2.5, n_rows), 0, 10)
    screen_time_hours = np.clip(rng.normal(7, 2.5, n_rows), 1, 15)
    caffeine_intake_mg = np.clip(rng.normal(180, 90, n_rows), 0, 500)

    job_satisfaction = rng.choice(["Low", "Medium", "High"], size=n_rows, p=[0.25, 0.45, 0.30])
    family_history = rng.choice(["Yes", "No"], size=n_rows, p=[0.30, 0.70])
    smoking_status = rng.choice(["Never", "Former", "Current"], size=n_rows, p=[0.60, 0.20, 0.20])
    gender = rng.choice(["Female", "Male", "Other"], size=n_rows, p=[0.48, 0.48, 0.04])

    # Construct latent risk score with intuitive relationships.
    risk_score = (
        0.32 * stress_level
        + 0.22 * (10 - sleep_hours)
        + 0.16 * (work_hours_per_week / 10)
        + 0.12 * (screen_time_hours / 2)
        + 0.15 * (10 - social_interaction_score)
        + 0.11 * (caffeine_intake_mg / 100)
        - 0.25 * (physical_activity_hours / 2)
        + np.where(job_satisfaction == "Low", 1.2, np.where(job_satisfaction == "Medium", 0.4, -0.5))
        + np.where(family_history == "Yes", 0.8, 0.0)
        + np.where(smoking_status == "Current", 0.6, np.where(smoking_status == "Former", 0.2, 0.0))
        + rng.normal(0, 0.9, n_rows)
    )

    # Convert score to 3-class target.
    bins = np.quantile(risk_score, [0.33, 0.66])
    risk_level = np.where(risk_score <= bins[0], "Low", np.where(risk_score <= bins[1], "Medium", "High"))

    df = pd.DataFrame(
        {
            "age": age,
            "work_hours_per_week": np.round(work_hours_per_week, 2),
            "sleep_hours": np.round(sleep_hours, 2),
            "stress_level": np.round(stress_level, 2),
            "physical_activity_hours": np.round(physical_activity_hours, 2),
            "social_interaction_score": np.round(social_interaction_score, 2),
            "screen_time_hours": np.round(screen_time_hours, 2),
            "caffeine_intake_mg": np.round(caffeine_intake_mg, 2),
            "job_satisfaction": job_satisfaction,
            "family_history": family_history,
            "smoking_status": smoking_status,
            "gender": gender,
            "risk_level": risk_level,
        }
    )

    # Inject missing values (about 3% in selected columns).
    cols_for_missing = [
        "sleep_hours",
        "stress_level",
        "social_interaction_score",
        "job_satisfaction",
        "screen_time_hours",
    ]
    for col in cols_for_missing:
        idx = rng.choice(df.index, size=max(1, int(0.03 * n_rows)), replace=False)
        df.loc[idx, col] = np.nan

    # Inject outliers in a small subset for cleaning demo.
    outlier_idx = rng.choice(df.index, size=max(1, int(0.015 * n_rows)), replace=False)
    df.loc[outlier_idx, "work_hours_per_week"] *= rng.uniform(1.5, 2.2, size=len(outlier_idx))
    df.loc[outlier_idx, "caffeine_intake_mg"] *= rng.uniform(1.4, 2.0, size=len(outlier_idx))
    df["work_hours_per_week"] = np.round(df["work_hours_per_week"], 2)
    df["caffeine_intake_mg"] = np.round(df["caffeine_intake_mg"], 2)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic mental health dataset CSV.")
    parser.add_argument("--rows", type=int, default=1200, help="Number of rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=str,
        default="mental_health_data.csv",
        help="Output CSV file path.",
    )
    args = parser.parse_args()

    df = generate_synthetic_mental_health_data(n_rows=args.rows, seed=args.seed)
    df.to_csv(args.output, index=False)

    print(f"Generated dataset: {args.output}")
    print(f"Shape: {df.shape}")
    print("Class distribution:")
    print(df["risk_level"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
