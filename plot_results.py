import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(ax, filename, label):
    with np.load(filename) as data:
        rewards = data["rewards"]

    block_size = 10000
    rounds = []
    accuracy = []

    # calc accuracy for each block of 10,000 rounds
    for start in range(0, len(rewards), block_size):
        end = min(start + block_size, len(rewards))
        rounds.append(end)
        accuracy.append(rewards[start:end].mean())

    ax.plot(rounds, accuracy, label=label)

def main():
    fig, axes = plt.subplots(1, 2, figsize=(12,4), sharey=True)

    # stationary experiments
    plot_rewards(
        axes[0],
        "NeuralBandit2_rounds6000000_drift0_seed67.npz",
        "NB2",
    )
    plot_rewards(
        axes[0],
        "NeuralBandit3_rounds6000000_drift0_seed67.npz",
        "NB3",
    )
    axes[0].set_title("stationary")

    # label shifts
    plot_rewards(
        axes[1],
        "NeuralBandit2_rounds1400000_drift500000_seed67.npz",
        "NB2",
    )
    plot_rewards(
        axes[1],
        "NeuralBandit3_rounds1400000_drift500000_seed67.npz",
        "NB3",
    )
    axes[1].set_title("label shifts")

    for shift in [500_000, 1_000_000]:
        axes[1].axvline(shift, color="gray", linestyle="--", alpha=0.7)

    for ax in axes:
        ax.set_xlabel("round")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("accuracy per 10,000 rounds")
    fig.suptitle("NeuralBandit results, seed 67")
    fig.tight_layout()
    fig.savefig("neuralbandit_results.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()