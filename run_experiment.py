import os
import numpy as np
import torch
from baselines import LinUCB, CTS, Banditron
from load_dataset import encode_features, load_dataset

def run_experiment(features, labels, row_order, model, rounds, seed, drift):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # start at random row in shuffled data
    start = rng.integers(len(row_order))
    bandit = model(rng=rng)
    rewards = np.zeros(rounds, dtype=np.uint8)

    for x in range(rounds):
        # loop back to start when we reach end
        row = row_order[(start + x) % len(row_order)]
        context = features[row]
        label = int(labels[row])

        # rotate correct class after each drift interval
        if drift > 0:
            shift = x // drift
            label = (label - 1 + shift) % 7 + 1

        action, prob = bandit.choose_action(context)
        reward = int(action == label)
        rewards[x] = reward
        bandit.update(context, action, reward, prob)

        if (x + 1) % 10_000 == 0:
            print(f"finished round: {x + 1:,} / {rounds:,}", flush=True,)

    return rewards


def main():
    features, labels = load_dataset()
    features = torch.from_numpy(encode_features(features))

    # use same shuffled order as neural experiments
    row_order = np.random.default_rng(67).permutation(len(labels))

    experiments = [(Banditron, 1_400_000, 500_000), (Banditron, 6_000_000, 0),]
    
    for model, rounds, drift in experiments:
        for seed in range(67, 77):
            filename = (
                f"{model.__name__}_rounds{rounds}"
                f"_drift{drift}_seed{seed}.npz"
            )

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

            window = min(100_000, len(rewards))
            print(
                f"last {window:,} avg reward: "
                f"{rewards[-window:].mean():.4f}",
                flush=True,
            )
            print(f"saved {filename}", flush=True)


if __name__ == "__main__":
    main()