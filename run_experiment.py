import os
import numpy as np
import torch

from baselines import LinUCB, CTS, Banditron
from neural_bandit import NeuralBandit2, NeuralBandit3
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
            print(
                f"finished round: {x + 1:,} / {rounds:,}",
                flush=True,
            )

    return rewards


def main():
    features, labels = load_dataset()
    features = torch.from_numpy(encode_features(features))

    # use the same shuffled order as before
    row_order = np.random.default_rng(67).permutation(len(labels))

    models = [NeuralBandit2, NeuralBandit3, LinUCB, CTS, Banditron]
    rounds = 2500000

    for drift in [0, 500000]:
        for model in models:
            for seed in range(67, 77):
                filename = (
                    f"{model.__name__}_rounds{rounds}"
                    f"_drift{drift}_seed{seed}.npz"
                )

                if os.path.exists(filename):
                    with np.load(filename) as data:
                        if data["rewards"].shape != (rounds,):
                            raise ValueError(f"incomplete run in {filename}")

                    print(f"skipping {filename}", flush=True)
                    continue

                old_filename = (
                    f"{model.__name__}_rounds6000000"
                    f"_drift0_seed{seed}.npz"
                )

                # reuse the first 2.5 million stationary rewards
                if drift == 0 and os.path.exists(old_filename):
                    with np.load(old_filename) as data:
                        old_rewards = data["rewards"]

                        if old_rewards.shape != (6_000_000,):
                            raise ValueError(
                                f"incomplete run in {old_filename}"
                            )

                        rewards = old_rewards[:rounds].copy()

                    print(f"reusing {old_filename}", flush=True)

                else:
                    print(
                        f"starting {model.__name__}, "
                        f"drift {drift}, seed {seed}",
                        flush=True,
                    )

                    rewards = run_experiment(
                        features, labels, row_order,
                        model, rounds, seed, drift,
                    )

                np.savez_compressed(filename, rewards=rewards)

                print(
                    f"last 100,000 avg reward: "
                    f"{rewards[-100_000:].mean():.4f}",
                    flush=True,
                )
                print(f"saved {filename}", flush=True)


if __name__ == "__main__":
    main()