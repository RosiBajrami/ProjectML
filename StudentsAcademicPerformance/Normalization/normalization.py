import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Shared categorical columns to exclude from normalization
categorical_columns = [
    'Daytime/evening attendance\t',
    'Displaced',
    'Debtor',
    'Tuition fees up to date',
    'Scholarship holder',
    'Gender_1',
    'Application mode_2',
    'Application mode_5',
    'Application mode_7',
    'Application mode_10',
    'Application mode_15',
    'Application mode_16',
    'Application mode_17',
    'Application mode_18',
    'Application mode_26',
    'Application mode_27',
    'Application mode_39',
    'Application mode_42',
    'Application mode_43',
    'Application mode_44',
    'Application mode_51',
    'Application mode_53',
    'Application mode_57',
    'Marital status_2',
    'Marital status_3',
    'Marital status_4',
    'Marital status_5',
    'Marital status_6'
]

# normalization min-max, decimal and z-score
def normalize_and_save(df_path, out_prefix):
    df = pd.read_csv(df_path)
    target = df.pop("Target")
    numeric_cols = df.columns.difference(categorical_columns)

    # Min-Max
    df_minmax = df.copy()
    df_minmax[numeric_cols] = MinMaxScaler().fit_transform(df_minmax[numeric_cols])
    df_minmax["Target"] = target
    df_minmax.to_csv(f"../Normalized_Datasets/{out_prefix}_minmax.csv", index=False)

    # Z-Score
    df_zscore = df.copy()
    df_zscore[numeric_cols] = StandardScaler().fit_transform(df_zscore[numeric_cols])
    df_zscore["Target"] = target
    df_zscore.to_csv(f"../Normalized_Datasets/{out_prefix}_zscore.csv", index=False)

    # Decimal Scaling
    df_decimal = df.copy()
    for col in numeric_cols:
        max_val = df_decimal[col].abs().max()
        scaling_factor = 10 ** len(str(int(max_val))) if max_val != 0 else 1
        df_decimal[col] = df_decimal[col] / scaling_factor
    df_decimal["Target"] = target
    df_decimal.to_csv(f"../Normalized_Datasets/{out_prefix}_decimal.csv", index=False)

    print(f"✅ Normalization complete for {out_prefix}")


# apply and save
normalize_and_save("../Preprocessed_Dataset/preprocessed_data.csv", "preprocessed")
normalize_and_save("../FeatureEngineered_Dataset/feature_engineered_data.csv", "fe")
