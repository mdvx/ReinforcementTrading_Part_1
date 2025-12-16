import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv

def main():
    df = load_and_preprocess_data("data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv")
    
    # create env
    env = ForexTradingEnv(df=df,
                          window_size=30,
                          sl_options=[30, 60, 80],  # example SL distances in pips
                          tp_options=[30, 60, 80])  # example TP distances in pips
    
    # Wrap in a DummyVecEnv (required by stable-baselines for parallelization)
    vec_env = DummyVecEnv([lambda: env])
    
    # Define RL model (PPO)
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_log/"
    )
    
    # Train the model
    model.learn(total_timesteps=10100)
    model.save("model_eurusd")
    print("Model saved successfully!")
    
    # Evaluate or test the model
    obs = vec_env.reset()
    done = False
    equity_curve = []
    
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = vec_env.step(action)
        
        # Collect equity from the unwrapped environment
        # Because we have a DummyVecEnv, we can access env_method to get the attribute
        current_equity = vec_env.get_attr("equity")[0]
        equity_curve.append(current_equity)
        
        if done[0]:
            break
    
    # Plot the final equity curve
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label='Equity')
    plt.title("Equity Curve during Evaluation")
    plt.xlabel("Time Steps")
    plt.ylabel("Equity")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
