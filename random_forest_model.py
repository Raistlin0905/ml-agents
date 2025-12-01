from typing import List, Tuple
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

from Model import Model

MIN_SAMPLES_RECOMMENDED = 30

# cols to exclude from features (identifiers and raw timestamps)
EXCLUDE_COLUMNS = ["run_id", "time_start", "time_end"]

TARGET_COLUMN = "total_duration"

# categorical cols that need encoding
CATEGORICAL_COLUMNS = [
    "cpu_vendor", "cpu_architecture", "ram_type", "trainer_type",
    "host_os_name", "disk_filesystem", "vis_encode_type",
    "gpu_0_name", "gpu_0_vendor", "gpu_1_name", "gpu_1_vendor",
    "learning_rate_schedule", "beta_schedule", "epsilon_schedule",
    "memory", "goal_conditioning_type", "init_path"
]

# bool cols to convert to 0/1
BOOLEAN_COLUMNS = [
    "is_docker_used", "normalize", "deterministic", "shared_critic",
    "even_checkpoints", "threaded", "self_play", "behavioral_cloning"
]


class RandomForestModel(Model[list[float], float]):
    """
    Random Forest Regressor for predicting ML-Agents training times
    """

    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            **kwargs
        )


    def train(self, training_data: List[Tuple[list[float], float]]) -> None:
        """
        Format of data: [(features, training_time), ...]
        Example: [([55, 4, 4, 104, 10000], 360.5), ([26, 2, 3, 500, 500000], 1800.2)]
        """
        n_samples = len(training_data)
        if n_samples < MIN_SAMPLES_RECOMMENDED:
            print(f"WARNING: Only {n_samples} samples. Recommended minimum: {MIN_SAMPLES_RECOMMENDED}")
            print("Model may be unreliable with limited data.")

        features, labels = zip(*training_data)
        self.model.fit(features, labels)


    def predict(self, x: List[list[float]]) -> List[float]:
        predictions = self.model.predict(x)
        return predictions.tolist()

    def train_from_csv(self, csv_path: str) -> None:
        """Load CSV and train in one step."""
        data = self.load_from_csv(csv_path)
        self.train(data)

    @staticmethod
    def load_from_csv(csv_path: str) -> List[Tuple[list[float], float]]:
        """
        Load and preprocess training data from CSV
        Returns a list of tuples ready for the training
        """
        df = pd.read_csv(csv_path)

        # separate target from features
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in CSV")

        target = df[TARGET_COLUMN]

        # remove excluded cols and target
        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS + [TARGET_COLUMN]]
        features_df = df[feature_cols].copy()

        # handle null values
        for col in features_df.columns:
            if col in CATEGORICAL_COLUMNS:
                # fill empty strings and nulls with "none"
                features_df[col] = features_df[col].fillna("none").replace("", "none")
            elif col in BOOLEAN_COLUMNS:
                # fill nulls with False
                features_df[col] = features_df[col].fillna(False)
            else:
                # numeric columns: fill with 0
                features_df[col] = features_df[col].fillna(0)

        # encode categorical columns
        for col in CATEGORICAL_COLUMNS:
            if col in features_df.columns:
                features_df[col] = pd.factorize(features_df[col].astype(str))[0]

        # convert boolean columns to 0/1
        for col in BOOLEAN_COLUMNS:
            if col in features_df.columns:
                features_df[col] = features_df[col].map(
                    lambda x: 1 if x in [True, "True", "true", 1, "1"] else 0
                )

        # convert to list of tuples
        training_data = []
        for idx in range(len(features_df)):
            features = features_df.iloc[idx].tolist()
            label = float(target.iloc[idx])
            training_data.append((features, label))

        return training_data

