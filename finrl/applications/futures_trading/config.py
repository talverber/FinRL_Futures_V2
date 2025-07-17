"""Global configuration for training and backtesting scripts."""

def load_namespace():
    import sys
    import os
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory
    gparent_dir = os.path.abspath(os.path.join(current_dir, '../../..'))

    # Add the parent directory to sys.path
    sys.path.append(gparent_dir)

    # Now you can import modules from the parent directory
    #import parent_module_name

load_namespace()

# Data folder to use for training and backtesting
# Possible options include 'futures_data' or 'retail_data'
from datetime import datetime
import multiprocessing
import numpy as np

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from finrl.meta.env_stock_trading.env_futurestrading import FuturesTradingEnv
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.main import check_and_make_directories

# =========================
# Configuration
# =========================
# DATA_TYPE is defined in config.py

DATA_TYPE = 'futures' #'futures' # 'futures' or 

ENV_TYPE = 'stocks'  # 'futures' or 'stocks'


if ENV_TYPE == 'futures' :
    # Use our attempt to implement [48, Zihao Zhang et al’].
    TradingEnv = FuturesTradingEnv
else:
    TradingEnv = StockTradingEnv



if ENV_TYPE == 'retail':
    INDICATORS = ['rmean_7', 'rmean_30', 'vix']

elif ENV_TYPE == 'futures':
    INDICATORS = ['ret','ret_1M','ret_2M','ret_3M','ret_1Y','macd', 'rsi_30']

elif ENV_TYPE == 'stocks':
    from finrl.config import INDICATORS as STOCK_INDICATORS
    INDICATORS = STOCK_INDICATORS #['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']
# not what we need for futures
else:
    raise NotImplementedError("unknown DATA_TYPE: " + DATA_TYPE)

#TSTP = datetime.now().strftime("%Y%m%d-%H%M") # Timestamp of the run 
TSTP = "20250717-1745"

check_and_make_directories([f'data/{DATA_TYPE}',])
TRAINED_MODEL_DIR = f'data/{DATA_TYPE}/trained_models/{TSTP}'
RESULTS_DIR       = f'data/{DATA_TYPE}/results/{TSTP}'
TRAIN_FILE        = f'data/{DATA_TYPE}/train_data.csv'  # Preprocessed training CSV
BACKTEST_FILE     = f'data/{DATA_TYPE}/trade_data.csv'
RAW_DATA_FILE     = f'data/{DATA_TYPE}/raw_data.csv' 



TIC_TO_USE        = None
# TIC_TO_USE = ['AAPL', 'INTC'] # Tickers/symbols to include
# If None, use entire dataset

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE   = '2020-07-01'
TRADE_START_DATE = '2020-07-01'
TRADE_END_DATE   = '2021-10-29'
# TRADE_START_DATE = '2015-01-24' # '2020-07-01'
# TRADE_END_DATE = '2015-07-31' # '2021-10-29'


# Trading environment parameters
COST_PCT       = 0.001       # 0.1% cost
HMAX           = 100         # max shares per trade
INITIAL_AMOUNT = 1_000_000   # starting cash
REWARD_SCALING = 1e-4        # reward scale
TURB_THRESHOLD  = 70
RISK_COL        = 'vix'


# Number of parallel envs to use
N_ENVS    = min(8, multiprocessing.cpu_count())   # cap per‑algo envs
MAX_PROCS = 2                                     # cap # concurrent algos


# Plot dimensions (width, height) in inches
PLOT_SIZE       = (15, 5)


ALGOS_TO_USE = ['ppo', 'ddpg', 'a2c', 'td3', 'sac'] # Algorithms to train

# Map algorithm names to their classes
MODEL_CLASSES = {
    'a2c': A2C,
    'ddpg': DDPG,
    'ppo': PPO,
    'td3': TD3,
    'sac': SAC,
}

# Total timesteps for each algorithm
TOTAL_TIMESTEPS = {
    'a2c':  50_000,
    'ddpg': 50_000,
    'ppo': 200_000,
    'td3':  50_000,
    'sac':  70_000,
}

# Hyperparameter overrides per algorithm
HYPERPARAMS = {
    #
    # ►  A2C  ◄
    #
    "a2c": {
        "n_steps":       5 * 256,     # rollout length (multiple of 5)
        "gamma":         0.99,        # discount
        "learning_rate": 7e-4,        # Adam LR in the paper
        "ent_coef":      0.01,        # entropy bonus
    },
    #
    # ►  DDPG  ◄
    #
    "ddpg": {
        "batch_size":    128,
        "buffer_size":   1_000_000,   # replay buffer
        "learning_rate": 1e-3,        # actor & critic
        "tau":           0.005,       # soft-update speed
        "gamma":         0.99,
    },
    #
    # ►  PPO  ◄
    #
    "ppo": {
        "n_steps":       2048,
        "batch_size":    128,
        "ent_coef":      0.01,
        "learning_rate": 2.5e-4,
        "device": "cpu"
    },
    #
    # ►  TD3  ◄
    #
    "td3": {
        "batch_size":    100,
        "buffer_size":   1_000_000,
        "learning_rate": 1e-3,
    },
    #
    # ►  SAC  ◄
    #
    "sac": {
        "batch_size":    128,
        "buffer_size":   100_000,
        "learning_rate": 1e-4,
        "learning_starts": 100,
        "ent_coef":      "auto_0.1",
    },
}
