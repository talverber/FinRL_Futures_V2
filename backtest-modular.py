import os
import sys
import subprocess
from pathlib import Path
import empyrical as ep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from pypfopt.efficient_frontier import EfficientFrontier

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_futurestrading import FuturesTradingEnv
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

# Hyperparameters and configuration
# DATA_TYPE is defined in config.py
# Possible values include 'futures_data' or 'retail_data'
from config import DATA_TYPE, INDICATORS

TSTP = "20250706-1505"

ALGOS_TO_USE    = ['ppo', 'a2c',  'ddpg', 'td3', 'sac']
TRAIN_FILE      = f'{DATA_TYPE}/train_data.csv'
BACKTEST_FILE   = f'{DATA_TYPE}/trade_data.csv'
TRAINED_MODEL_DIR = f'{DATA_TYPE}/trained_models/{TSTP}'


INITIAL_AMOUNT  = 1_000_000
COST_PCT        = 0.001
HMAX            = 100
REWARD_SCALING  = 1e-4
TURB_THRESHOLD  = 70
RISK_COL        = 'vix'
OUTPUT_PLOT     = f'{DATA_TYPE}/results/{TSTP}/backtest.png'
# Plot dimensions (width, height) in inches
PLOT_SIZE       = (15, 5)

TRADE_START_DATE = '2020-07-01'
TRADE_END_DATE = '2021-10-29'
# TRADE_START_DATE = '2015-01-24' # '2020-07-01'
# TRADE_END_DATE = '2015-07-31' # '2021-10-29'

TradingEnv = FuturesTradingEnv  # StockTradingEnv  #

# Map algorithm names to their classes
MODEL_CLASSES = {
    'a2c': A2C,
    'ddpg': DDPG,
    'ppo': PPO,
    'td3': TD3,
    'sac': SAC,
}

def load_and_filter_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])

    # rebuild integer day index
    df = df.reset_index(drop=True)
    day_map = {d: i for i, d in enumerate(sorted(df['date'].unique()))}
    df['day'] = df['date'].map(day_map)
    return df.set_index(['day','tic'], drop=False)


def load_trained_models(algos: list) -> dict:
    models = {}
    for algo in algos:
        model_path = Path(TRAINED_MODEL_DIR) / f"agent_{algo}"
        try:
            models[algo] = MODEL_CLASSES[algo].load(str(model_path))
        except Exception:
            models[algo] = None
    return models


def run_drl_backtests(models: dict, env: gym.Env) -> dict:
    results = {}
    for algo, model in models.items():
        if model is not None:
            df_account, _ = DRLAgent.DRL_prediction(model=model, environment=env)
            df_account = df_account.set_index(df_account.columns[0])
            results[algo] = df_account['account_value']
        else:
            results[algo] = None
    return results


def compute_mvo(train_df: pd.DataFrame, trade_df: pd.DataFrame) -> pd.Series:
    # Pivot to wide form
    price_train = train_df.pivot(index='date', columns='tic', values='close')
    price_trade = trade_df.pivot(index='date', columns='tic', values='close')
    # Daily returns (%)
    returns_train = price_train.pct_change().dropna() * 100
    # Mean and covariance
    mean_returns = returns_train.mean()
    cov_returns  = returns_train.cov()
    # Optimize weights
    ef = EfficientFrontier(mean_returns, cov_returns, weight_bounds=(0, 0.5))
    ef.max_sharpe()
    cleaned_w = ef.clean_weights()
    # Convert weights to share counts
    last_prices = price_train.iloc[-1]
    init_shares = {
        tic: (INITIAL_AMOUNT * cleaned_w[tic]) / last_prices[tic]
        for tic in cleaned_w
    }
    init_shares = pd.Series(init_shares)
    # Backtest on trading data
    portfolio_vals = price_trade.dot(init_shares)
    return pd.Series(portfolio_vals.values, index=price_trade.index, name='Mean Var')


def fetch_dji_series() -> pd.Series:
    df_dji = YahooDownloader(
        start_date=TRADE_START_DATE,
        end_date=TRADE_END_DATE,
        ticker_list=['dji']
    ).fetch_data()
    df_dji = df_dji[['date', 'close']]
    df_dji.date = pd.to_datetime(df_dji.date)
    first = df_dji['close'].iloc[0]
    scaled = df_dji['close'].div(first).mul(INITIAL_AMOUNT)
    return pd.Series(scaled.values, index=df_dji['date'], name='dji')


def plot_and_save(results: pd.DataFrame, filename: str):
    results.plot(figsize=PLOT_SIZE)
    # plt.tight_layout()
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.title('Backtest: DRL vs MVO vs DJIA')
    plt.savefig(filename, dpi=150)
    # Auto-open based on OS
    if sys.platform == 'darwin':
        subprocess.Popen(['open', filename])
    elif sys.platform.startswith('linux'):
        subprocess.Popen(['xdg-open', filename])
    elif sys.platform.startswith('win'):
        os.startfile(filename)


def main():
    plt.ion()
    # Load datasets
    train = load_and_filter_data(TRAIN_FILE)
    trade = load_and_filter_data(BACKTEST_FILE)
    print('TRAIN head:\n', train.head(), '\n')
    print('TRADE head:\n', trade.head(), '\n')
    print('Shapes:', train.shape, trade.shape)

    # Load DRL models
    models = load_trained_models(ALGOS_TO_USE)

    # Prepare trading environment
    stock_dim = len(trade['tic'].unique())
    state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
    print(f"Stock Dimension: {stock_dim}, State Space: {state_space}")

    env_kwargs = {
        'hmax': HMAX,
        'initial_amount': INITIAL_AMOUNT,
        'num_stock_shares': [0] * stock_dim,
        'buy_cost_pct': [COST_PCT] * stock_dim,
        'sell_cost_pct': [COST_PCT] * stock_dim,
        'state_space': state_space,
        'stock_dim': stock_dim,
        'tech_indicator_list': INDICATORS,
        'action_space': stock_dim,
        'reward_scaling': REWARD_SCALING
    }
    env = TradingEnv(
        df=trade,
        turbulence_threshold=TURB_THRESHOLD,
        risk_indicator_col=RISK_COL,
        **env_kwargs
    )

    # Compute MVO and DJIA baselines
    mvo_series = compute_mvo(train, trade)
    dji_series = fetch_dji_series()

    # Run DRL backtests
    drl_results = run_drl_backtests(models, env)

    result = pd.DataFrame({
        'a2c': drl_results.get('a2c'),
        'ddpg': drl_results.get('ddpg'),
        'ppo': drl_results.get('ppo'),
        'td3': drl_results.get('td3'),
        'sac': drl_results.get('sac'),
        'mvo': mvo_series,
        'dji': dji_series,
    })

    result.to_csv(f'{DATA_TYPE}/results/{TSTP}/backtest.csv')

    metrics = {}
    for col in result.columns:
        series = result[col].dropna()
        if len(series) < 2:
            continue                               # ignore empty curves
        rets = series.pct_change().dropna()
        metrics[col] = {
            "Final value"  : series.iloc[-1],
            "Ann. return"  : ep.annual_return(rets),
            "Ann. std"     : ep.annual_volatility(rets),
            "Sharpe"       : ep.sharpe_ratio(rets),
            "Max drawdown" : ep.max_drawdown(rets),
        }

    df_metrics = pd.DataFrame(metrics).T
    fpath = f"{DATA_TYPE}/results/{TSTP}/backtest_metrics.csv"
    df_metrics.to_csv(fpath, float_format="%.6f")
    print(f"✔ metrics written → {fpath}")
    plot_and_save(result, OUTPUT_PLOT)

if __name__ == '__main__':
    main()
