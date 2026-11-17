import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class MarketTradingEnv(gym.Env):
    """
    Gymnasium-compatible Algorithmic Trading Environment with RNN Market State Representation.
    
    Observation Space: Box(low=-inf, high=inf, shape=(49,), dtype=float32)
      - [0:32]  : 32-D market state embedding from pre-trained frozen LSTM encoder
      - [32:46] : 14-D explicit normalized technical features at close of day t
      - [46:49] : 3-D internal portfolio state at close of day t
                  - 46: Current equity position fraction (shares * Close_t / V_t) in [0, 1]
                  - 47: Available cash ratio (cash / V_t) in [0, 1]
                  - 48: Normalized portfolio value (V_t / initial_capital)
                  
    Action Space: Discrete(3)
      - 0: HOLD
      - 1: BUY  (Deploy available cash to buy maximum integer shares at Open_{t+1})
      - 2: SELL (Liquidate all held shares at Open_{t+1})
      
    Timing Convention:
      - At day t close: Observation o_t is generated.
      - Action a_t is selected.
      - At day t+1 open: Trade executes at Open_{t+1} with transaction fee c = 0.1%.
      - At day t+1 close: Portfolio is marked to market at Close_{t+1}.
      - Reward r_{t+1} = log(V_{t+1} / V_t).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        embeddings: torch.Tensor,
        features: torch.Tensor,
        metadata_df: pd.DataFrame,
        raw_prices_dict: dict,
        initial_capital: float = 100000.0,
        transaction_fee: float = 0.001,
        symbol: str = None
    ):
        super().__init__()
        
        self.initial_capital = float(initial_capital)
        self.transaction_fee = float(transaction_fee)
        self.symbol = symbol
        
        # Filter metadata by symbol if specified
        if symbol is not None:
            mask = (metadata_df["Symbol"] == symbol).values
            self.metadata_df = metadata_df[mask].reset_index(drop=True)
            self.embeddings = embeddings[mask].float().numpy()
            self.features = features[mask, -1, :].float().numpy() # 14-D at timestep -1
        else:
            self.metadata_df = metadata_df.reset_index(drop=True)
            self.embeddings = embeddings.float().numpy()
            self.features = features[:, -1, :].float().numpy()
            
        self.raw_prices = raw_prices_dict
        self.num_samples = len(self.metadata_df)
        
        # Precompute price sequences to avoid dataframe lookups during stepping
        self.symbols = self.metadata_df["Symbol"].values
        self.end_dates = self.metadata_df["Window_End_Date"].values
        self.target_dates = self.metadata_df["Target_Date"].values
        
        self.close_t_arr = np.zeros(self.num_samples, dtype=np.float64)
        self.open_t1_arr = np.zeros(self.num_samples, dtype=np.float64)
        self.close_t1_arr = np.zeros(self.num_samples, dtype=np.float64)
        
        for idx in range(self.num_samples):
            sym = self.symbols[idx]
            raw_df = self.raw_prices[sym]
            d_t = self.end_dates[idx]
            d_t1 = self.target_dates[idx]
            
            # Lookup prices
            row_t = raw_df[raw_df["Date"] == d_t]
            row_t1 = raw_df[raw_df["Date"] == d_t1]
            
            self.close_t_arr[idx] = row_t["Close"].values[0]
            self.open_t1_arr[idx] = row_t1["Open"].values[0]
            self.close_t1_arr[idx] = row_t1["Close"].values[0]

        # Gymnasium Spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(49,), dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.prev_portfolio_value = self.initial_capital
        self.history = []

    def _get_observation(self) -> np.ndarray:
        idx = self.current_step
        emb = self.embeddings[idx] # 32-D
        feat = self.features[idx]   # 14-D
        
        # Portfolio state at close of day t
        close_t = self.close_t_arr[idx]
        equity_val = self.shares * close_t
        curr_val = self.cash + equity_val
        
        if curr_val <= 1e-8:
            pos_ratio = 0.0
            cash_ratio = 0.0
            norm_val = 0.0
        else:
            pos_ratio = float(equity_val / curr_val)
            cash_ratio = float(self.cash / curr_val)
            norm_val = float(curr_val / self.initial_capital)
            
        port_state = np.array([pos_ratio, cash_ratio, norm_val], dtype=np.float32)
        obs = np.concatenate([emb, feat, port_state]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.prev_portfolio_value = self.initial_capital
        self.history = []
        
        obs = self._get_observation()
        info = {
            "step": self.current_step,
            "date": self.end_dates[0],
            "target_date": self.target_dates[0],
            "symbol": self.symbols[0],
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "shares": self.shares,
            "current_price": self.close_t_arr[0]
        }
        return obs, info

    def step(self, action: int):
        idx = self.current_step
        close_t = self.close_t_arr[idx]
        open_t1 = self.open_t1_arr[idx]
        close_t1 = self.close_t1_arr[idx]
        
        # Portfolio value before trading day t+1
        v_prev = self.cash + (self.shares * close_t)
        self.prev_portfolio_value = v_prev
        
        trade_executed = False
        trade_shares = 0
        trade_price = open_t1
        cost = 0.0
        
        # 1. Action Execution at Open_{t+1}
        if action == 1: # BUY
            # Only buy if not currently holding shares (prevents unintended repeated churn)
            if self.shares == 0:
                max_spendable_shares = int(self.cash // (open_t1 * (1.0 + self.transaction_fee)))
                if max_spendable_shares > 0:
                    trade_shares = max_spendable_shares
                    trade_val = trade_shares * open_t1
                    cost = trade_val * self.transaction_fee
                    self.cash -= (trade_val + cost)
                    self.shares += trade_shares
                    trade_executed = True
                
        elif action == 2: # SELL
            if self.shares > 0:
                trade_shares = self.shares
                trade_val = trade_shares * open_t1
                cost = trade_val * self.transaction_fee
                self.cash += (trade_val - cost)
                self.shares = 0
                trade_executed = True
                
        # 2. Mark portfolio to market at Close_{t+1}
        v_curr = self.cash + (self.shares * close_t1)
        self.portfolio_value = v_curr
        
        # 3. Calculate log return reward
        if v_prev > 1e-8 and v_curr > 1e-8:
            reward = float(np.log(v_curr / v_prev))
        else:
            reward = -10.0 # Extreme penalty if ruined
            
        # Record step history
        self.history.append({
            "step": idx,
            "date": self.target_dates[idx],
            "symbol": self.symbols[idx],
            "action": action,
            "trade_executed": trade_executed,
            "trade_shares": trade_shares,
            "trade_price": trade_price,
            "cost": cost,
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": v_curr,
            "reward": reward
        })
        
        self.current_step += 1
        terminated = (self.current_step >= self.num_samples)
        truncated = False
        
        if terminated:
            # Return last valid observation or terminal observation
            self.current_step -= 1
            obs = self._get_observation()
            self.current_step += 1
        else:
            obs = self._get_observation()
            
        info = {
            "step": self.current_step,
            "date": self.target_dates[idx],
            "symbol": self.symbols[idx],
            "portfolio_value": v_curr,
            "cash": self.cash,
            "shares": self.shares,
            "current_price": close_t1,
            "action_taken": action,
            "trade_executed": trade_executed,
            "trade_shares": trade_shares,
            "trade_price": trade_price,
            "transaction_cost": cost,
            "reward": reward
        }
        
        return obs, reward, terminated, truncated, info
