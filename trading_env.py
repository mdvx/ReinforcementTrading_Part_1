import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ForexTradingEnv(gym.Env):
    """
    A custom trading environment for EUR/USD that at each step:
      - Observes a window of data (OHLC + indicators).
      - Takes an action: choose among NO TRADE or combos of (direction, SL, TP).
      - Computes reward based on PnL from that decision.
    """
    
    def __init__(self, df, window_size=30, sl_options=None, tp_options=None):
        super(ForexTradingEnv, self).__init__()
        
        # Store the dataframe containing prices and indicators
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        
        # Observation parameters
        self.window_size = window_size
        
        # Discretize SL and TP distances in pips or in price terms
        # e.g. [10, 20, 30] pips from entry
        self.sl_options = sl_options if sl_options else [60, 90, 120]
        self.tp_options = tp_options if tp_options else [60, 90, 120]
        
        # --- Construct a discrete action space with NO TRADE option ---
        # Action 0 => No Trade
        # Then for direction in [0=short, 1=long] and each sl, tp => next actions
        self.action_map = [(None, None, None)]  # (None, None, None) => no trade
        for direction in [0, 1]:  # 0=short, 1=long
            for sl in self.sl_options:
                for tp in self.tp_options:
                    self.action_map.append((direction, sl, tp))

        # The total number of discrete actions is 1 + 2 * len(sl_options) * len(tp_options)
        self.action_space = spaces.Discrete(len(self.action_map))
        
        # Number of features in the observation
        self.num_features = self.df.shape[1]

        # We'll return a window of these features as a 2D array
        # shape = (window_size, num_features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, self.num_features), dtype=np.float32
        )
        
        # Internal state
        self.current_step = 0
        self.done = False
        self.equity = 10000.0  # starting capital
        self.max_slippage = 0.000  # example slippage
        self.positions = []  # track open positions if you want; or one position at a time
        
        # For logging
        self.equity_curve = []
        self.last_trade_info = None  # track the last trade details
    
    def _get_observation(self):
        """
        Returns the last 'window_size' observations as a 2D numpy array of shape (window_size, num_features).
        If at the start (not enough history), pad with the earliest row or zeros.
        """
        start = max(self.current_step - self.window_size, 0)
        obs_df = self.df.iloc[start:self.current_step]
        
        # If there's not enough data to fill 'window_size', pad with the earliest row
        if len(obs_df) < self.window_size:
            padding_rows = self.window_size - len(obs_df)
            first_part = np.tile(obs_df.iloc[0].values, (padding_rows, 1))
            obs_array = np.concatenate([first_part, obs_df.values], axis=0)
        else:
            obs_array = obs_df.values
        
        return obs_array.astype(np.float32)
    
    def _calculate_reward(self, direction, sl, tp):
        """
        A very simplified approach:
        - Immediately calculate PnL based on next bar's movement (or multiple bars) until SL/TP is hit.
        - In a real scenario, you'd keep the position open for multiple steps.
        """
        entry_price = self.df.loc[self.current_step, "Close"]
        
        # If last step, no movement
        if self.current_step >= self.n_steps - 1:
            return 0.0
        
        next_high = self.df.loc[self.current_step + 1, "High"]
        next_low = self.df.loc[self.current_step + 1, "Low"]
        
        # Convert pips to price distance
        pip_value = 0.0001
        sl_price_distance = sl * pip_value
        tp_price_distance = tp * pip_value
        
        # direction=1 => Long, direction=0 => Short
        if direction == 1:
            stop_loss = entry_price - sl_price_distance
            take_profit = entry_price + tp_price_distance
            
            # check if next_low < stop_loss => SL triggered
            # check if next_high > take_profit => TP triggered
            if next_low <= stop_loss and next_high >= take_profit:
                # if both SL and TP are touched assume it's a loss
                pnl = -sl_price_distance
            elif next_low <= stop_loss:
                pnl = -sl_price_distance
            elif next_high >= take_profit:
                pnl = tp_price_distance
            else:
                # use close for partial reward
                next_close = self.df.loc[self.current_step + 1, "Close"]
                pnl = next_close - entry_price
        else:
            # direction=0 => Short
            stop_loss = entry_price + sl_price_distance
            take_profit = entry_price - tp_price_distance
            
            if next_high >= stop_loss and next_low <= take_profit:
                if (stop_loss - entry_price) < (entry_price - take_profit):
                    pnl = -sl_price_distance
                else:
                    pnl = tp_price_distance
            elif next_high >= stop_loss:
                pnl = -sl_price_distance
            elif next_low <= take_profit:
                pnl = tp_price_distance
            else:
                next_close = self.df.loc[self.current_step + 1, "Close"]
                pnl = entry_price - next_close
        
        # reward in "pips" => multiply by 10,000 to convert from price difference
        reward = pnl * 10000
        return reward

    def step(self, action):
        """
        action is an integer in [0..(1 + 2*len(SL)*len(TP))-1],
        where:
          0 => do nothing
          else => (direction, sl, tp)
        """
        # Decode the action
        direction, sl, tp = self.action_map[action]

        if direction is None:
            reward = 0.0
            exit_price = None
            self.last_trade_info = {
                "entry_price": None,
                "exit_price": None,
                "pnl": 0.0
            }
        else:
            entry_price = self.df.loc[self.current_step, "Close"]
            reward = self._calculate_reward(direction, sl, tp)

            if self.current_step < self.n_steps - 1:
                exit_price = self.df.loc[self.current_step + 1, "Close"]
            else:
                exit_price = entry_price

            self.last_trade_info = {
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": reward / 10000.0
            }

            self.equity += reward

        self.equity_curve.append(self.equity)

        # Advance time
        self.current_step += 1

        # Episode end conditions
        truncated = (self.current_step >= self.n_steps - 1)  # end of dataset / time limit
        terminated = (self.equity <= 0.0)  # “real” terminal condition (optional)

        self.done = terminated or truncated  # if you still use self.done elsewhere

        # Next observation (if you prefer, you can clamp step before building obs)
        obs = self._get_observation()

        info = {
            "equity": self.equity,
            "last_trade": self.last_trade_info,
        }

        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)   # important: sets up self.np_random

        self.current_step = self.window_size  # start so we have a full window
        self.equity = 10000.0
        self.done = False
        self.equity_curve = []
        self.last_trade_info = None
        info = {"equity": self.equity}

        return self._get_observation(), info
    
    def render(self, mode='human'):
        """Optional: print or plot debug info."""
        print(f"Step: {self.current_step}, Equity: {self.equity}")
