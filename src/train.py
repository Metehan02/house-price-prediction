import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.preprocessing import create_preprocessor, load_data, separate_target


def evaluate_model_cv(model, X, y, cv):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    rmse_scores = -scores
    return rmse_scores.mean(), rmse_scores.std()


def main():
    train_path = "data/raw/train.csv"
    test_path = "data/raw/test.csv"

    train_df, _ = load_data(train_path, test_path)
    X, y = separate_target(train_df)

    y = np.log1p(y)

    preprocessor = create_preprocessor(X)

    models = {
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    for name, regressor in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

        mean_rmse, std_rmse = evaluate_model_cv(pipeline, X, y, cv)

        results.append({
            "Model": name,
            "CV Mean RMSE": mean_rmse,
            "CV Std RMSE": std_rmse
        })

    results_df = pd.DataFrame(results).sort_values("CV Mean RMSE").reset_index(drop=True)

    print("\nCross-Validation Model Comparison:")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_rmse = results_df.iloc[0]["CV Mean RMSE"]

    print(f"\nBest Model: {best_model_name}")
    print(f"Best CV Mean RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    main()