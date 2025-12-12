import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import DQN

from rsaenv import RSAEnv


# Utilities

def make_env(capacity: int, request_files: List[str], seed: int = 0) -> Monitor:
    """
    Helper to build a monitored RSAEnv instance.
    """
    env = RSAEnv(capacity=capacity, request_files=request_files, seed=seed)
    env = Monitor(env)  # tracks episode rewards/lengths
    env.reset(seed=seed)
    return env


def moving_averages(
    episode_rewards: List[float],
    episode_length: int,
    window: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute moving averages of (episode reward, blocking rate) over the last
    `window` episodes.

    Episode blocking rate B is computed as:
        B = 1 - (episode_reward / episode_length)
    because reward = 1 for accepted requests and 0 otherwise.
    """
    n = len(episode_rewards)
    if n == 0:
        return np.array([]), np.array([])

    window = min(window, n)
    avg_reward = np.zeros(n, dtype=np.float32)
    avg_blocking = np.zeros(n, dtype=np.float32)

    for i in range(n):
        start = max(0, i - window + 1)
        r = np.mean(episode_rewards[start : i + 1])
        avg_reward[i] = r
        avg_blocking[i] = 1.0 - (r / float(episode_length))

    return avg_reward, avg_blocking


def plot_curve(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    path: str,
) -> None:
    """
    Save a simple line plot of y vs x.
    """
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved plot to: {path}")


# Training and evaluation for a given capacity

def train_dqn_for_capacity(
    capacity: int,
    train_files: List[str],
    n_episodes: int = 500,
    episode_length: int = 100,
    seed: int = 0,
    out_dir: str = "results",
    learning_rate: float = 1e-3,
    batch_size: int = 64,
) -> str:
    """
    Train a DQN agent for a given link capacity and return the model path.

    This function:
      1. Creates a single RSAEnv wrapped with Monitor.
      2. Trains DQN for `n_episodes * episode_length` time steps.
      3. Extracts episode rewards from the Monitor.
      4. Computes moving averages of rewards and blocking rates.
      5. Saves two plots for this capacity:
           - learning curve (avg episode reward vs. episode)
           - avg blocking rate vs. episode
    """
    os.makedirs(out_dir, exist_ok=True)

    set_random_seed(seed)

    # Environment and sanity check 
    raw_env = RSAEnv(capacity=capacity, request_files=train_files, seed=seed)
    check_env(raw_env, warn=True)
    raw_env.close()

    env = make_env(capacity, train_files, seed=seed)

    model_name = f"dqn_rsa_cap{capacity}"
    model_path = os.path.join(out_dir, model_name)

    # Configure DQN
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=batch_size,
        gamma=0.99,
        target_update_interval=1_000,
        train_freq=4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.3,
        verbose=1,
        seed=seed,
    )

    # Large total_timesteps; each episode has exactly `episode_length` requests.
    total_timesteps = n_episodes * episode_length
    env.reset()
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save trained model
    model.save(model_path)
    print(f"Saved model to: {model_path}.zip")

    # Extract per-episode rewards from Monitor
    episode_rewards = env.get_episode_rewards()
    n_eps = len(episode_rewards)
    print(f"Training finished after {n_eps} episodes.")

    # Compute moving averages for last-10-episodes window
    avg_reward, avg_blocking = moving_averages(
        episode_rewards, episode_length=episode_length, window=10
    )
    episodes_axis = np.arange(1, n_eps + 1)

    # Plot learning curve: avg reward vs episode
    reward_plot_path = os.path.join(out_dir, f"cap{capacity}_learning_curve.png")
    plot_curve(
        x=episodes_axis,
        y=avg_reward,
        xlabel="Episode",
        ylabel="Average episode reward (last 10 episodes)",
        title=f"Capacity {capacity}: Learning curve",
        path=reward_plot_path,
    )

    # Plot blocking rate vs episode
    blocking_plot_path = os.path.join(out_dir, f"cap{capacity}_blocking_train.png")
    plot_curve(
        x=episodes_axis,
        y=avg_blocking,
        xlabel="Episode",
        ylabel="Average blocking rate (last 10 episodes)",
        title=f"Capacity {capacity}: Blocking rate (training)",
        path=blocking_plot_path,
    )

    env.close()
    return model_path


def evaluate_dqn_on_eval_set(
    model_path: str,
    capacity: int,
    eval_files: List[str],
    episode_length: int = 100,
    out_dir: str = "results",
) -> None:
    """
    Run the saved DQN model on the eval dataset using deterministic policy.

    Each eval CSV file corresponds to one episode. We compute the blocking
    rate per episode and then plot the moving average (window size up to 10).
    """
    from stable_baselines3 import DQN as DQN_cls

    model = DQN_cls.load(model_path)

    episode_rewards = []
    for idx, req_file in enumerate(eval_files):
        env = RSAEnv(capacity=capacity, request_files=[req_file])
        obs, info = env.reset(options={"request_file": req_file})
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        episode_rewards.append(ep_reward)
        env.close()
        print(f"[Eval] Capacity {capacity}, episode {idx+1}, reward = {ep_reward}")

    n_eps = len(episode_rewards)
    avg_reward, avg_blocking = moving_averages(
        episode_rewards, episode_length=episode_length, window=10
    )
    episodes_axis = np.arange(1, n_eps + 1)

    blocking_plot_path = os.path.join(out_dir, f"cap{capacity}_blocking_eval.png")
    plot_curve(
        x=episodes_axis,
        y=avg_blocking,
        xlabel="Episode (eval file)",
        ylabel="Average blocking rate (last 10 episodes)",
        title=f"Capacity {capacity}: Blocking rate (eval set)",
        path=blocking_plot_path,
    )


# Main 

if __name__ == "__main__":
    # Paths to training and eval request CSVs 
    base_data_dir = os.path.join(os.path.dirname(__file__), "data")
    train_dir = os.path.join(base_data_dir, "train")
    eval_dir = os.path.join(base_data_dir, "eval")

    train_files = sorted(
        [
            os.path.join(train_dir, f)
            for f in os.listdir(train_dir)
            if f.endswith(".csv")
        ]
    )
    eval_files = sorted(
        [
            os.path.join(eval_dir, f)
            for f in os.listdir(eval_dir)
            if f.endswith(".csv")
        ]
    )

    # Hyperparameters 
    SEED = 123
    N_EPISODES = 500           # number of training episodes per capacity
    EPISODE_LEN = 100          # each request file has 100 requests
    OUT_DIR = "results"

    # Part 1: capacity = 20 
    cap20_model_path = train_dqn_for_capacity(
        capacity=20,
        train_files=train_files,
        n_episodes=N_EPISODES,
        episode_length=EPISODE_LEN,
        seed=SEED,
        out_dir=OUT_DIR,
        learning_rate=1e-3,
        batch_size=64,
    )
    evaluate_dqn_on_eval_set(
        model_path=cap20_model_path,
        capacity=20,
        eval_files=eval_files,
        episode_length=EPISODE_LEN,
        out_dir=OUT_DIR,
    )

    # Part 2: capacity = 10
    cap10_model_path = train_dqn_for_capacity(
        capacity=10,
        train_files=train_files,
        n_episodes=N_EPISODES,
        episode_length=EPISODE_LEN,
        seed=SEED + 1,
        out_dir=OUT_DIR,
        learning_rate=5e-4,
        batch_size=128,
    )
    evaluate_dqn_on_eval_set(
        model_path=cap10_model_path,
        capacity=10,
        eval_files=eval_files,
        episode_length=EPISODE_LEN,
        out_dir=OUT_DIR,
    )
