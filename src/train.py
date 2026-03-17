import numpy as np

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import create_preprocessor, load_data, separate_target


def evaluate_model(name, model, X_train, X_valid, y_train, y_valid):
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, predictions))
    print(f"{name} RMSE: {rmse:.4f}")
    return rmse


def main():
    train_path = "data/raw/train.csv"
    test_path = "data/raw/test.csv"

    train_df, _ = load_data(train_path, test_path)
    X, y = separate_target(train_df)

    y = np.log1p(y)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = create_preprocessor(X)

    models = {
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    for name, regressor in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

        evaluate_model(name, pipeline, X_train, X_valid, y_train, y_valid)


if __name__ == "__main__":
    main()