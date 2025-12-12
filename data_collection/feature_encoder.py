import sys
import pandas

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


EXCLUDED_COLUMNS = ["run_id"]


def encode_and_save(input_path, output_path, class_label):
    # Load human-readable csv file
    try:
        dataframe = pandas.read_csv(input_path)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    # Store run_id before dropping (keeping for traceability)
    run_id = dataframe["run_id"] if "run_id" in dataframe.columns else None

    # Drop excluded columns
    dataframe = dataframe.drop(
        [col for col in EXCLUDED_COLUMNS if col in dataframe.columns.tolist()], axis=1
    )

    # Identify class label and features
    Y = dataframe[class_label]
    X = dataframe.drop(columns=[class_label])

    # Handle missing values safely (AFTER splitting X/Y)
    for col in X.columns:
        if X[col].dtype in ["int64", "float64"]:
            X[col] = X[col].fillna(0)
        else:
            X[col] = X[col].fillna("NaN")

    # Get text and numeric columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    # Normalize numerical cols using min/max and mult by 100 for prettiness
    if numeric_cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols]) * 100

    X_parts = []
    feature_names = []  # For preserving feature-names

    if numeric_cols:
        X_parts.append(X[numeric_cols])
        feature_names.extend(numeric_cols)

    # One-Hot Encoding
    if categorical_cols:
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        matrix = one_hot_encoder.fit_transform(X[categorical_cols])
        one_hot_feature_names = one_hot_encoder.get_feature_names_out(categorical_cols)
        one_hot_dataframe = pandas.DataFrame(
            matrix, columns=one_hot_feature_names, index=X.index
        )

        # Add encoded col and feature names
        X_parts.append(one_hot_dataframe)
        feature_names.extend(one_hot_feature_names)

    # Put together encoded cols and numeric cols
    if X_parts:
        X_final = pandas.concat(X_parts, axis=1)
    else:
        X_final = pandas.DataFrame(index=X.index)

    # Add run_id
    if run_id is not None:
        X_final.insert(0, "run_id", run_id)
    # Add class label
    X_final[class_label] = Y

    # Save everything to one csv
    X_final.to_csv(output_path + ".csv", index=False)


def print_usage_and_exit():
    print(
        "\nUsage:\n"
        "  !!!WORKING DIRECTORY IS ROOT DIRECTORY!!!\n"
        "  python data_collection/feature_encoder.py <input_path> <output_path>\n\n"
        "Arguments:\n"
        "  input_path    Path to the .csv dataset file to be encoded (path/to/input.csv)\n"
        "  output_path   Path of the output file to be created (path/to/output.csv)\n"
        "  class_label   Column name of class label (e.g., time_to_threshold)\n",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 4:
        print_usage_and_exit()

    input_path = args[1]
    output_path = args[2]
    class_label = args[3]

    encode_and_save(input_path, output_path, class_label)
