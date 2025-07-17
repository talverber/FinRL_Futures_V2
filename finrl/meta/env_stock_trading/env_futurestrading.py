from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

matplotlib.use("Agg")

# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat


class FuturesTradingEnv(StockTradingEnv):
    """
    A stock trading environment for OpenAI gym

    Parameters:
        df (pandas.DataFrame): Dataframe containing data
        hmax (int): Maximum cash to be traded in each trade per asset.
        initial_amount (int): Amount of cash initially available
        buy_cost_pct (float, array): Cost for buying shares, each index corresponds to each asset
        sell_cost_pct (float, array): Cost for selling shares, each index corresponds to each asset
        turbulence_threshold (float): Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated
        print_verbosity(int): When iterating (step), how often to print stats about state of env
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.periods = ['ret_1Y', 'ret_3M', 'ret_2M', 'ret_1M'] #['ret_1M', 'ret_2M', 'ret_3M', 'ret_1Y']
        self.days = [252, 63, 42, 21]
        self.time_scale = pd.DataFrame([self.days], columns = self.periods)

        
        # self.df = self.df[['day', 'tic', 'close', 'macd', 'rsi_30', 
        #                    'volatility', 'ret'] + self.periods ]

        self.df[['ret'] + self.periods] = 0.0
        self.tracking = self.df.loc[0,self.periods ]

    def step(self, actions):
        self.terminal = self.day >= self.df.index.get_level_values('day').nunique() - 1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.asset_memory[0]
            )  # initial_amount is only cash part of our initial asset
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, False, {}

        else:
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = actions.astype(
                int
            )  # convert into integer because we can't by fraction of shares
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)

            begin_balance = self.state[0]

            # Entry Price * Number of Contracts
            begin_value = np.array(self.state[1 : (self.stock_dim + 1)])\
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1

            self.data = self.df.loc[self.day, :]
            upd = self.calc_periods_returns()
            self.data.loc[:,self.periods] = upd.loc[self.day]


            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_balance = self.state[0]
            
            # Exit Price * Number of Contracts
            end_value = np.array(self.state[1 : (self.stock_dim + 1)])\
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])


            mu = 1 # a.k.a. "Contract Unit" or "Contract Multiplier"
            transaction_cost_term = 0 # 
            sigma_tgt = 15

            PnL = end_value - begin_value # a.k.a. returns

            self.df.loc[self.day, "ret"] = np.array(PnL)

            sigmas = np.array(self.data.volatility)
       
            rewards = mu * (actions * PnL * sigma_tgt / sigmas - transaction_cost_term)


            self.asset_memory.append(end_balance + sum(end_value))
            self.date_memory.append(self._get_date())
            self.reward = sum(rewards)/self.stock_dim
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step

        return self.state, self.reward, self.terminal, False, {}


    def calc_periods_returns(self):

        s = np.array(self.days) 
        s = np.maximum(self.day - s, np.array([0]))  

        volatility = self.df.loc[(self.day, slice(None)), ['volatility']]
        volatility.columns = pd.RangeIndex(start=0, stop=1, step=1)
        norm = 1 / np.sqrt(volatility @ self.time_scale)
               
               
        addition = self.df.loc[self.day - 1,['ret','ret','ret','ret']] 
        addition.columns = self.periods       
        
        
        tmp = {k:v for k,v in zip(s,self.periods)}
        lower = self.df.loc[(s, slice(None)),'ret']
        lower = lower.unstack(level=0, )
        lower.columns = lower.columns.map(tmp)


        addition -= lower
        addition.fillna(0.0, inplace= True)
        self.tracking += addition
        update = self.tracking * norm
 
        return update