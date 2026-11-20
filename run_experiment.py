import numpy as np
import torch
from load_dataset import encode_features, load_dataset
from neural_bandit import NeuralBandit2, NeuralBandit3
import os

def run_experiment(features, labels, row_order, model, rounds, seed, drift):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # start at random row in the shuffled dataset
    start = rng.integers(len(row_order))
    bandit = model(rng=rng)
    rewards = np.zeros(rounds, dtype=np.uint8)

    for x in range(rounds):
        # loop back to start when we reach end
        row = row_order[(start + x) % len(row_order)]
        context = features[row]
        label = int(labels[row])

        # after every drift interval, rotate the correct class 
        if drift > 0:
            shift = x // drift
            label = (label - 1 + shift) % 7 + 1

        action, prob = bandit.choose_action(context)
        reward = int(action == label)
        rewards[x] = reward
        bandit.update(context, action, reward, prob)

        if (x + 1) % 10000 == 0:
            print(f"finished round: {x + 1:,} / {rounds:,}", flush=True)

    return rewards

def main():
    features, labels = load_dataset()
    features = torch.from_numpy(encode_features(features))
    row_order = np.random.default_rng(67).permutation(len(labels))

    experiments = [
        (NeuralBandit2, 1_400_000, 500_000),
        (NeuralBandit3, 1_400_000, 500_000),
        (NeuralBandit2, 6_000_000, 0),
        (NeuralBandit3, 6_000_000, 0),
    ]

    for model, rounds, drift in experiments:
        for seed in range(67, 77):
            filename = (
                f"{model.__name__}_rounds{rounds}"
                f"_drift{drift}_seed{seed}.npz"
            )

            # skip runs that already have saved results
            if os.path.exists(filename):
                print(f"skipping {filename}", flush=True)
                continue

            print(
                f"starting {model.__name__}, drift {drift}, seed {seed}",
                flush=True,
            )

            rewards = run_experiment(
                features, labels, row_order, model, rounds, seed, drift
            )

            np.savez_compressed(filename, rewards=rewards)
            print(
                f"last 100,000 avg reward: {rewards[-100_000:].mean():.4f}",
                flush=True,
            )
            print(f"saved {filename}", flush=True)

if __name__ == "__main__":
    main()