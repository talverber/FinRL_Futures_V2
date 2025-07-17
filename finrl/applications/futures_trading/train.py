import os
import pandas as pd


from config import  TRAIN_FILE, TRAINED_MODEL_DIR, RESULTS_DIR, \
    ALGOS_TO_USE, INDICATORS, INITIAL_AMOUNT, COST_PCT, HMAX, REWARD_SCALING, TradingEnv,\
    BACKTEST_FILE, TRADE_END_DATE, TRADE_START_DATE, MODEL_CLASSES, TURB_THRESHOLD, PLOT_SIZE, RISK_COL, \
    TRAIN_START_DATE, TRAIN_END_DATE, HYPERPARAMS, N_ENVS, TIC_TO_USE, TOTAL_TIMESTEPS

#from concurrent.futures import ProcessPoolExecutor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.main import check_and_make_directories
#from config import *

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
        return TradingEnv(
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
    return TradingEnv(
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
