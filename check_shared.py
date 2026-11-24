import numpy as np
import torch
from load_digits_dataset import load_digits_dataset
from seasonal_recall_shared import SeasonalRecallShared

def main():
    rng = np.random.default_rng(83)
    torch.manual_seed(83)

    features, labels = load_digits_dataset()

    bandit = SeasonalRecallShared(
        rng=rng,
        input_size=64,
        num_actions=10,
    )

    bandit.archive_interval = 100
    bandit.signature_size = 16

    for step in range(500):
        context = torch.from_numpy(features[step])
        label = int(labels[step])

        action, prob = bandit.choose_action(context)

        assert 1 <= action <= 10
        assert 0 < prob <= 1

        reward = int(action == label)
        bandit.update(context, action, reward, prob)

    assert 1 <= len(bandit.memory) <= 3
    assert np.isfinite(bandit.router_weights).all()
    assert np.isfinite(bandit.log_weights).all()

    print("digits action and probability checks passed")
    print("memory archive check passed")
    print("model selection weights are finite")

if __name__ == "__main__":
    main()