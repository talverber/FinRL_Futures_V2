import os
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_futurestrading import FuturesTradingEnv

from datetime import datetime
# =========================
# Configuration
# =========================
# DATA_TYPE is defined in config.py
from config import DATA_TYPE, INDICATORS

TSTP = datetime.now().strftime("%Y%m%d-%H%M")

TRAINED_MODEL_DIR = f'{DATA_TYPE}/trained_models/{TSTP}'
RESULTS_DIR = f'{DATA_TYPE}/results/{TSTP}'
TRAIN_FILE        = f'{DATA_TYPE}/train_data.csv'  # Preprocessed training CSV


ALGOS_TO_USE      = ['ppo', 'ddpg', 'a2c', 'td3', 'sac']               # Algorithms to train
TIC_TO_USE        = None
# TIC_TO_USE = ['AAPL', 'INTC'] # Tickers/symbols to include
# If None, use entire dataset
TRAIN_START_DATE  = None  # e.g. '2015-01-01'
TRAIN_END_DATE    = None  # e.g. '2015-12-31'


# Total timesteps for each algorithm
TOTAL_TIMESTEPS = {
    'a2c':  50_000,
   'ddpg':  50_000,
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

# Trading environment parameters
COST_PCT       = 0.001       # 0.1% cost
HMAX           = 100         # max shares per trade
INITIAL_AMOUNT = 1_000_000   # starting cash
REWARD_SCALING = 1e-4        # reward scale

# Number of parallel envs to use
N_ENVS    = min(14, multiprocessing.cpu_count())   # cap per‑algo envs
MAX_PROCS = 2                                     # cap # concurrent algos

def load_and_filter_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])

    # optional date slice
    if TRAIN_START_DATE and TRAIN_END_DATE:
        df = df[(df['date'] >= TRAIN_START_DATE) & (df['date'] <= TRAIN_END_DATE)]

    # optional ticker slice
    if TIC_TO_USE is not None:
        df = df[df['tic'].isin(TIC_TO_USE)].copy()

    # rebuild integer day index
    df = df.reset_index(drop=True)
    day_map = {d: i for i, d in enumerate(sorted(df['date'].unique()))}
    df['day'] = df['date'].map(day_map)
    return df.set_index(['day','tic'], drop=False)


def make_vec_env(df: pd.DataFrame) -> SubprocVecEnv:
    """
    Create a SubprocVecEnv of N_ENVS copies of StockTradingEnv.
    """
    def make_env_fn():
        return FuturesTradingEnv(
            df=df,
            hmax=HMAX,
            initial_amount=INITIAL_AMOUNT,
            num_stock_shares=[0]*df['tic'].nunique(),
            buy_cost_pct=[COST_PCT]*df['tic'].nunique(),
            sell_cost_pct=[COST_PCT]*df['tic'].nunique(),
            state_space=1 + 2*df['tic'].nunique() + df['tic'].nunique()*len(INDICATORS),
            stock_dim=df['tic'].nunique(),
            tech_indicator_list=INDICATORS,
            action_space=df['tic'].nunique(),
            reward_scaling=REWARD_SCALING
        )
    return SubprocVecEnv([make_env_fn for _ in range(N_ENVS)])


def make_single_env(df):
    # for quick testing without parallelism
    return FuturesTradingEnv(
        df=df,
        hmax=HMAX,
        initial_amount=INITIAL_AMOUNT,
        num_stock_shares=[0]*df['tic'].nunique(),
        buy_cost_pct=[COST_PCT]*df['tic'].nunique(),
        sell_cost_pct=[COST_PCT]*df['tic'].nunique(),
        state_space=1 + 2*df['tic'].nunique() + df['tic'].nunique()*len(INDICATORS),
        stock_dim=df['tic'].nunique(),
        tech_indicator_list=INDICATORS,
        action_space=df['tic'].nunique(),
        reward_scaling=REWARD_SCALING
    )

def train_and_save(algo: str):
    """
    Train a single algorithm using parallel envs.
    """
    df = load_and_filter_data(TRAIN_FILE)
    vec_env = make_vec_env(df)
    #vec_env = make_single_env(df) for debugging without multiprocessing

    agent = DRLAgent(env=vec_env)
    model = agent.get_model(algo, model_kwargs=HYPERPARAMS.get(algo, {}))

    # Configure logger
    log_path = os.path.join(RESULTS_DIR, algo)
    new_logger = configure(log_path, ['stdout', 'csv', 'tensorboard'])
    model.set_logger(new_logger)

    # Train
    timesteps = TOTAL_TIMESTEPS[algo]
    trained = agent.train_model(
        model=model,
        tb_log_name=algo,
        total_timesteps=timesteps
    )

    # Save
    save_path = os.path.join(TRAINED_MODEL_DIR, f'agent_{algo}')
    trained.save(save_path)
    print(f"Saved {algo} model to {save_path}")


# =========================
# Main execution
# =========================
def main():
    check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR])
    for algo in ALGOS_TO_USE:
        try:
            train_and_save(algo)
        except Exception as e:
            print(f"❌ Error training {algo}: {e}")
            break

if __name__ == '__main__':
    print(os.getcwd())
    main()
