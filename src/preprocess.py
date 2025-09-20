import pandas as pd
import os
from sklearn.model_selection import train_test_split

# paths
RAW_DATA = "data/data-1754297123597.csv"
PROCESSED_DIR = "data/processed"
TRAIN_FILE = os.path.join(PROCESSED_DIR, "train.csv")
TEST_FILE = os.path.join(PROCESSED_DIR, "test.csv")

# features and target
FEATURES = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes', 'controller_kind']
TARGET = 'cpu_usage'

def main():
    # load
    df = pd.read_csv(RAW_DATA)

    # handle missing controller_kind
    df['controller_kind'] = df['controller_kind'].fillna("Unknown")

    # select relevant columns
    df = df[FEATURES + [TARGET]]

    # train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # save processed data
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print(f"✅ Preprocessing complete: {TRAIN_FILE}, {TEST_FILE}")

if __name__ == "__main__":
    main()
