from random_forest_model import RandomForestModel
from sklearn.model_selection import train_test_split
import pandas as pd

CSV_PATH = "cleaned_pool.csv"
TARGET_COLUMN = "total_duration"


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples from {CSV_PATH}")

    numeric_df = df.select_dtypes(include=["number"])
    print(f"Using {len(numeric_df.columns) - 1} numeric features")

    X = numeric_df.drop(columns=[TARGET_COLUMN])
    y = numeric_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestModel()
    training_data = list(zip(X_train.values.tolist(), y_train.tolist()))
    model.train(training_data)
    model.feature_columns = X.columns.tolist()
    print("Model trained")

    predictions = model.predict(X_test.values.tolist())

    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\nResults on test set ({len(y_test)} samples):")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R^2:   {r2:.3f}")

    print("\nfeature importances:")
    importances = list(zip(model.feature_columns, model.model.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    for name, imp in importances[:10]:
        print(f"  {name}: {imp:.3f}")


if __name__ == "__main__":
    main()
