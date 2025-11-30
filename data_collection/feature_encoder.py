import sys
import pandas
import xgboost
from sklearn.model_selection import train_test_split


EXCLUDED_COLUMNS = [
    "run_id",
    "time_to_threshold_class_label",
    "behavior_name",
    "filesystem",
    "model",
]


def start(csv_path, class_label):
    try:
        data = pandas.read_csv(csv_path)

        run_id_col = "run_id" if "run_id" in data.columns else None
        run_ids = data["run_id"].copy() if run_id_col else None

        data = data.drop(
            [col for col in EXCLUDED_COLUMNS if col in data.columns], axis=1
        )

        X = data.drop(class_label, axis=1)
        y = data[class_label]
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        categorical_columns = X_train.select_dtypes(include=["object"]).columns
        for col in categorical_columns:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = (
                X_test[col]
                .astype("category")
                .cat.set_categories(X_train[col].cat.categories)
            )

        regressor = xgboost.XGBRegressor(enable_categorical=True)
        regressor.fit(X_train, y_train)

        encoded = X_train.copy()
        encoded[class_label] = y_train
        if run_id_col and run_ids is not None:
            encoded[run_id_col] = run_ids.loc[X_train.index]

        encoded.to_csv("feature_encoded.csv", index=False)

    except Exception as e:
        print(e, file=sys.stderr)


def print_usage_and_exit():
    print(
        "\nUsage:\n"
        "  !!!WORKING DIRECTORY IS ROOT DIRECTORY!!!\n"
        "  python data_collection/feature_encoder.py <path_to_dataset> <class_label_column_name>\n\n"
        "Arguments:\n"
        "  path_to_dataset              Path to .csv dataset file (e.g., human_readable.csv)\n"
        "  class_label_column_name      Name of the class label (target column; e.g., time_to_threshold)\n",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 3:
        print_usage_and_exit()

    start(args[1], args[2])
