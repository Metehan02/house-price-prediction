import os
import joblib
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import load_data, separate_target, create_preprocessor


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

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge())
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, predictions))

    print(f"Validation RMSE: {rmse:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/ridge_model.pkl")
    print("Model saved to models/ridge_model.pkl")


if __name__ == "__main__":
    main()