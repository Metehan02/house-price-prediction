import joblib
import numpy as np
import pandas as pd


def main():
    model = joblib.load("models/ridge_model.pkl")
    train_df = pd.read_csv("data/raw/train.csv")

    min_id = train_df["Id"].min()
    max_id = train_df["Id"].max()

    print(f"Choose a house ID (between {min_id} and {max_id}):")

    try:
        house_id = int(input("Enter house ID: "))
    except ValueError:
        print("Invalid input. Please enter a numeric house ID.")
        return

    selected_house = train_df[train_df["Id"] == house_id]

    if selected_house.empty:
        print("House ID not found.")
        return

    actual_price = selected_house["SalePrice"].values[0]
    features = selected_house.drop(columns=["SalePrice"])

    predicted_log_price = model.predict(features)[0]
    predicted_price = np.expm1(predicted_log_price)

    accuracy = 100 - (abs(actual_price - predicted_price) / actual_price * 100)

    print(f"Predicted price: ${predicted_price:,.2f}")
    print(f"Actual price: ${actual_price:,.2f}")
    print(f"Prediction accuracy: {accuracy:.2f}%")

    
if __name__ == "__main__":
    main()