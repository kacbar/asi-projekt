import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1, "Other": 2})
    df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map({"Yes": 1, "No": 0})
    df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map({"Yes": 1, "No": 0})
    df["Financial Stress"] = pd.to_numeric(df["Financial Stress"], errors="coerce")

    def parse_sleep(x):
        try:
            return float(x.split()[0])
        except:
            return None

    df["Sleep Duration"] = df["Sleep Duration"].apply(parse_sleep)

    print("\n🎯 Balans klas depresji:")
    print(df["Depression"].value_counts())
    print(df["Depression"].value_counts(normalize=True))

    return df
