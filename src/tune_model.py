import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from src.preprocessing import create_preprocessor, load_data, separate_target


def main():
    train_path = "data/raw/train.csv"
    test_path = "data/raw/test.csv"

    train_df, _ = load_data(train_path, test_path)
    X, y = separate_target(train_df)

    y = np.log1p(y)

    preprocessor = create_preprocessor(X)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(random_state=42))
    ])

    param_grid = {
        "regressor__n_estimators": [100, 200],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__max_depth": [3, 4],
        "regressor__subsample": [0.8, 1.0]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)

    best_rmse = -grid_search.best_score_

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print(f"\nBest CV RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    main()