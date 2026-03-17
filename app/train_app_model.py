import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def main():
    train_df = pd.read_csv("data/raw/train.csv")

    selected_features = [
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "TotalBsmtSF",
        "YearBuilt",
        "FullBath",
        "LotArea"
    ]

    X = train_df[selected_features]
    y = np.log1p(train_df["SalePrice"])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, predictions))

    print(f"App model RMSE: {rmse:.4f}")

    os.makedirs("app", exist_ok=True)
    joblib.dump(model, "app/app_model.pkl")
    print("App model saved to app/app_model.pkl")


if __name__ == "__main__":
    main()