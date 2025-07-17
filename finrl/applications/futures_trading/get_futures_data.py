import os
import pandas as pd
import yfinance as yf

from config import TRAIN_FILE, BACKTEST_FILE, INDICATORS, RAW_DATA_FILE, \
                   TRADE_START_DATE,TRADE_END_DATE, TRAIN_START_DATE, TRAIN_END_DATE

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

FUTURES_TICKERS = [
    'ES=F', 'CL=F', 'GC=F', 'NG=F', 'SI=F', 'ZC=F', 'ZS=F', 'ZW=F',
    'LE=F', 'HE=F', 'OJ=F', 'KC=F', 'SB=F', 'CC=F', 'CT=F', 'PL=F',
    'PA=F', 'RB=F', 'HO=F', 'M2K=F', 'MNQ=F', 'YM=F', 'HG=F', 'ZT=F',
    'ZN=F', 'ZB=F'
]

def download_futures_data(tickers, start, end):
    df_list = []
    for tic in tickers:
        df = yf.download(tic, start=start, end=end, auto_adjust=False)
        if df.empty:
            print(f"⚠️ Skipping {tic}: no data")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = {'Open','High','Low','Close','Volume'}
        if not required.issubset(df.columns):
            print(f"⚠️ Skipping {tic}: missing OHLCV")
            continue

        df = df.reset_index()
        df['tic'] = tic
        df = df[['Date','Open','High','Low','Close','Volume','tic']]
        df.columns = ['date','open','high','low','close','volume','tic']
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df_list.append(df)
    if not df_list:
        raise RuntimeError("No valid futures data fetched.")
    return pd.concat(df_list, ignore_index=True)

# STEP 1: Download

if not os.path.exists(RAW_DATA_FILE):
    print("Downloading futures data...")
    df_raw = download_futures_data(FUTURES_TICKERS, TRAIN_START_DATE, TRADE_END_DATE)

    df_raw.to_csv(RAW_DATA_FILE)
else: 
    print(f"Using cached raw data from {RAW_DATA_FILE}")
    df_raw = pd.read_csv(RAW_DATA_FILE)


# STEP 2: Add 'day' column
df_raw['day'] = pd.to_datetime(df_raw['date']).dt.dayofweek.astype(float)

# STEP 3: Feature engineering (include VIX and turbulence)
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=True,
    use_turbulence=True,
    user_defined_feature=False
)
processed = fe.preprocess_data(df_raw)

# STEP 4: Fill missing technical indicator columns with 0
expected_cols = [
    'date','tic','close','high','low','open','volume','day',
    'macd','boll_ub','boll_lb','rsi_30','cci_30','dx_30',
    'close_30_sma','close_60_sma','vix','turbulence'
]
for col in expected_cols:
    if col not in processed:
        processed[col] = 0.0

processed = processed[expected_cols]

# STEP 5: Sort so each date-tic is together (just for cleanliness)
processed = processed.sort_values(['date','tic']).reset_index(drop=True)


def add_volatility(df):
    wat = df.groupby("tic", as_index=True)['close']\
        .ewm(com=60, adjust=True).std()
    wat = wat.droplevel(0)
    df["volatility"] = wat
    return df

def add_returns(df):
    # make placeholders for returns data. to be calculated during training from actual returns
    inds = ['ret','ret_1M','ret_2M','ret_3M','ret_1Y']
    for i in inds:
        df[i] = 0.0
    return df

processed = add_volatility(processed)
processed = add_returns(processed)

# STEP 6: Split into train/trade
train = data_split(processed, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(processed, TRADE_START_DATE, TRADE_END_DATE)

# STEP 7: Rebuild date_idx for train and trade so both start from 0!
def reset_date_idx(df):
    df = df.copy()
    unique_dates = sorted(df['date'].unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    df['date_idx'] = df['date'].map(date_to_idx)
    return df

train = reset_date_idx(train)
trade = reset_date_idx(trade)

# STEP 8: Order columns like stock data (date_idx first, then others)
ordered_cols = [
    'date_idx','date','tic','close','high','low','open','volume','day',
    'macd','boll_ub','boll_lb','rsi_30','cci_30','dx_30',
    'close_30_sma','close_60_sma','vix','turbulence', 'volatility',
    'ret','ret_1M','ret_2M','ret_3M','ret_1Y'
]
train = train[ordered_cols]
trade = trade[ordered_cols]

# STEP 9: Save to CSV (no extra Pandas index)
train.to_csv(TRAIN_FILE, index=False)
trade.to_csv(BACKTEST_FILE, index=False)

print(f"✅ {TRAIN_FILE} and {BACKTEST_FILE} generated.")
