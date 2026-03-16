import joblib
import numpy as np
import pandas as pd


def main():
    model = joblib.load("models/ridge_model.pkl")
    test_df = pd.read_csv("data/raw/test.csv")

    single_house = test_df.iloc[[0]]

    prediction_log = model.predict(single_house)[0]
    prediction_price = np.expm1(prediction_log)

    print("House ID:", single_house["Id"].values[0])
    print(f"Predicted Sale Price: ${prediction_price:,.2f}")


if __name__ == "__main__":
    main()