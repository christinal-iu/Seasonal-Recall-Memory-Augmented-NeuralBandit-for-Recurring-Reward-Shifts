import numpy as np
import torch

from load_dataset import encode_features, load_dataset
from neural_bandit import NeuralBandit2

def main():
    rng = np.random.default_rng(67)
    torch.manual_seed(67)

    features, labels = load_dataset()
    features = torch.from_numpy(encode_features(features))

    # shuffle rows for a short stationary run
    rounds = 10000
    row_order = rng.permutation(len(labels))[:rounds]
    rewards = np.zeros(rounds)

    bandit = NeuralBandit2(rng=rng)

    for x, row in enumerate(row_order):
        context = features[row]
        action, prob = bandit.choose_action(context)
        reward = int(labels[row] == action)
        rewards[x] = reward

        # give bandit only the reward for the action it chose
        bandit.update(context, action, reward, prob)

    print(f"rounds: {rounds}")
    print(f"avg reward: {rewards.mean():.4f}")

if __name__ == "__main__":
    main()