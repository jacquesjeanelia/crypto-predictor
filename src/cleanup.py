from huggingface_hub import hf_hub_download, login, HfFileSystem, HfApi
import pandas as pd
import numpy as np

def get_dfs(total: bool = False):
    # Download the files from Hugging Face Hub and read them into a single DataFrame
    print("Downloading files from Hugging Face Hub...")
    token = "YOUR_HUGGINGFACE_TOKEN"  # Replace with your actual Hugging Face token
    repo_id = "zongowo111/v2-crypto-ohlcv-data"

    api = HfApi(token=token)

    all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    parquet_files = [f for f in all_files if f.endswith('.parquet')]

    dfs_1min = []
    dfs_15min = []
    dfs_1hour = []
    dfs_1day = []
    dfs = []

    if total:
        for file_path in parquet_files:
            parts = file_path.split('/')
            symbol = parts[-2] if "klines" in parts else parts[-1].split('.')[0]

            local_path = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset", token=token)
            temp_df = pd.read_parquet(local_path)
            temp_df['symbol'] = symbol
            print(symbol)
            dfs.append(temp_df)
        df = pd.concat(dfs, ignore_index=True)
    else:
        for file_path in parquet_files:
            parts = file_path.split('/')
            if "1m" in parts[-1]:
                symbol = parts[-2] if "klines" in parts else parts[-1].split('.')[0]
                local_path = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset", token=token)
                temp1_df = pd.read_parquet(local_path)
                temp1_df['symbol'] = symbol
                print(symbol, "1min")
                dfs_1min.append(temp1_df)
            elif "15" in parts[-1]:
                symbol = parts[-2] if "klines" in parts else parts[-1].split('.')[0]
                local_path = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset", token=token)
                temp15_df = pd.read_parquet(local_path)
                temp15_df['symbol'] = symbol
                # print(symbol, "15min")
                dfs_15min.append(temp15_df)
            elif "1h" in parts[-1]:
                symbol = parts[-2] if "klines" in parts else parts[-1].split('.')[0]
                local_path = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset", token=token)
                temp_hour_df = pd.read_parquet(local_path)
                temp_hour_df['symbol'] = symbol
                # print(symbol, "1h")
                dfs_1hour.append(temp_hour_df)
            elif "1d" in parts[-1]:
                symbol = parts[-2] if "klines" in parts else parts[-1].split('.')[0]
                local_path = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset", token=token)
                temp_day_df = pd.read_parquet(local_path)
                temp_day_df['symbol'] = symbol
                # print(symbol, "1d")
                dfs_1day.append(temp_day_df)
        df_1min = pd.concat(dfs_1min, ignore_index=True)
        df_15min = pd.concat(dfs_15min, ignore_index=True) 
        df_1hour = pd.concat(dfs_1hour, ignore_index=True)
        df_1day = pd.concat(dfs_1day, ignore_index=True)

    return df if total else [df_1min, df_15min, df_1hour, df_1day]

def cleanup(df, csv_name: str = "crypto_ohlcv_data_total_cleaned"):
    
    columns = [
        'open_time',
        'open', 
        'high', 
        'low', 
        'close', 
        'volume', 
        'close_time', 
        'quote_asset_volume', 
        'number_of_trades', 
        'taker_buy_base_asset_volume', 
        'taker_buy_quote_asset_volume', 
        'ignore'
    ]

    # Make sure all columns are numeric
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clean price columns by replacing 0 with NaN and forward filling missing values within each symbol group
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df.groupby('symbol')[col].ffill()

    df = df.drop(columns=['ignore', 'close_time'])

    print("Sort by time and set index to open_time")
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.sort_values(by=['symbol', 'open_time']).reset_index(drop=True)
    df = df.set_index('open_time')

    print("Calculating additional features")
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    df['returns'] = (df['close'] - df['open']) / df['open']
    df['taker_buy_ratio'] = df['taker_buy_base_asset_volume'] / df['volume']
    df['taker_buy_ratio'] = df['taker_buy_ratio'].replace([np.inf, -np.inf], pd.NA)
    df['taker_buy_ratio'] = df['taker_buy_ratio'].fillna(0.5)
    df['volatility'] = (df['high'] - df['low']) / df['close']
    df['body_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9) 

    grouped = df.groupby('symbol')

    # Calculate rolling mean and standard deviation for volume and volatility
    mean_volume = grouped['volume'].transform(lambda x: x.rolling(20).mean())
    mean_volatility = grouped['volatility'].transform(lambda x: x.rolling(20).mean())
    std_volatility = grouped['volatility'].transform(lambda x: x.rolling(20).std())

    df['relative_volume'] = df['volume'] / mean_volume
    df['normalized_volatility'] = (df['volatility'] - mean_volatility) / std_volatility

    # Handle cases where mean_volume or std_volatility is zero to avoid division by zero
    df.loc[mean_volume == 0, 'relative_volume'] = 0.0
    df.loc[std_volatility == 0, 'normalized_volatility'] = 0.0

    # Create label: 1 if next close price is higher than current close price, else 0
    df['label'] = grouped['close'].transform(lambda x: (x.shift(-1)) > x).astype(int)

    print("Dropping rows with missing values")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    # df.to_csv(f'out/preprocessed/{csv_name}_preprocessed.csv', index=True)

    feature_columns = [
        'hour', 'day_of_week',
        'returns', 'volatility',
        'upper_wick', 'lower_wick', 
        'taker_buy_ratio', 'body_ratio', 
        'relative_volume', 'normalized_volatility'
    ]

    for col in feature_columns:
        lower_limit = df[col].quantile(0.001)
        upper_limit = df[col].quantile(0.999)
        df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)

    # For each symbol, split the dataset chronologically into training and testing sets (80% train, 20% test)
    train_list, test_list = [], []
    for symbol, group in df.groupby('symbol'):
        split_index = int(len(group) * 0.8)
        train_list.append(group.iloc[:split_index])
        test_list.append(group.iloc[split_index:])

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    X_train, y_train = train_df[feature_columns], train_df['label']
    X_test, y_test = test_df[feature_columns], test_df['label']

    df = df[feature_columns + ['label']]

    df.to_csv(f'out/processed/{csv_name}.csv', index=True)

def run():
    # Get all datapoints, sorted by open time
    total_df = get_dfs(True)
    cleanup(total_df, "crypto_ohlcv_data_total_cleaned")
    # Get data separated: 15min, 1h, 1d intervals
    distinct_dfs = get_dfs(False)
    cleanup(distinct_dfs[0], "crypto_ohlcv_data_1min_cleaned")
    cleanup(distinct_dfs[1], "crypto_ohlcv_data_15min_cleaned")
    cleanup(distinct_dfs[2], "crypto_ohlcv_data_1h_cleaned")
    cleanup(distinct_dfs[3], "crypto_ohlcv_data_1d_cleaned")

if __name__ == "__main__":
    run()