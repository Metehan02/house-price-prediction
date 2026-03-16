import joblib
import numpy as np
import pandas as pd

from src.preprocessing import load_data


def main():
    train_path = "data/raw/train.csv"
    test_path = "data/raw/test.csv"

    _, test_df = load_data(train_path, test_path)

    model = joblib.load("models/ridge_model.pkl")

    predictions = model.predict(test_df)

    predictions = np.expm1(predictions)

    submission = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": predictions
    })

    submission.to_csv("submissions/submission.csv", index=False)

    print("Predictions saved to submissions/submission.csv")


if __name__ == "__main__":
    main()