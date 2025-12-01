import json
import sys
import pandas

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack, save_npz


EXCLUDED_COLUMNS = ["run_id"]


def encode_and_save(input_path, output_path, class_label):
    # Load human-readable csv file
    try:
        dataframe = pandas.read_csv(input_path)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)

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
    text_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    categorical_cols = []
    free_text_cols = []

    for col in text_cols:
        unique_val_count = X[col].nunique()
        avg_val_length = X[col].astype(str).str.len().mean()
        if unique_val_count <= 100 and avg_val_length <= 60:
            categorical_cols.append(col)
        else:
            free_text_cols.append(col)

    sparse_parts = []  # Not using dense matrices because too much memory
    feature_names = []  # For preserving feature-names

    if numeric_cols:
        csr = csr_matrix(X[numeric_cols].values)
        sparse_parts.append(csr)
        feature_names.extend(numeric_cols)

    # One-Hot Encoding
    if categorical_cols:
        one_hot_encoder = OneHotEncoder()
        matrix = one_hot_encoder.fit_transform(X[categorical_cols])
        sparse_parts.append(matrix)
        feature_names.extend(one_hot_encoder.get_feature_names_out(categorical_cols))
    # TF-IDF Vectorizing
    if free_text_cols:
        for col in free_text_cols:
            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform(X[col].astype(str))
            sparse_parts.append(matrix)
            feature_names.extend(
                [
                    f"{col}__{category}"
                    for category in vectorizer.get_feature_names_out()
                ]
            )

    # Put together encoded cols and numeric cols
    X_final = hstack(sparse_parts).tocsr()

    # Save encoded features to npz
    save_npz(output_path + ".npz", X_final)
    # Save class label to csv
    Y.to_csv(output_path + "_class_label.csv", index=False)
    # Save feature names to json
    with open(output_path + "_features.json", "w") as f:
        json.dump(feature_names, f)


def print_usage_and_exit():
    print(
        "\nUsage:\n"
        "  !!!WORKING DIRECTORY IS ROOT DIRECTORY!!!\n"
        "  python data_collection/feature_encoder.py <input_path> <output_path>\n\n"
        "Arguments:\n"
        "  input_path    Path to the .csv dataset file to be encoded\n"
        "  output_path   Path prefix for output file to be created (output -> output.npz)\n"
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

    print("yuh")
    encode_and_save(input_path, output_path, class_label)
