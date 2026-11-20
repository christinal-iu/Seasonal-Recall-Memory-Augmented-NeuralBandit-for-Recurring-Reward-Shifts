import csv
import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(ax, model, rounds, drift):
    block_size = 10_000
    curves = []
    final_accuracy = []

    for seed in range(67, 77):
        filename = (
            f"{model}_rounds{rounds}"
            f"_drift{drift}_seed{seed}.npz"
        )

        with np.load(filename) as data:
            rewards = data["rewards"]

        # check that the saved run is complete
        if rewards.shape != (rounds,):
            raise ValueError(f"unexpected reward shape in {filename}")

        # average each block within this run
        curve = rewards.reshape(-1, block_size).mean(axis=1)
        curves.append(curve)
        final_accuracy.append(rewards[-100_000:].mean())

    # calculate variation across the ten runs
    curves = np.array(curves)
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0, ddof=1)
    round_numbers = np.arange(block_size, rounds + 1, block_size)

    line, = ax.plot(round_numbers, mean_curve, label=model)
    ax.fill_between(
        round_numbers,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color=line.get_color(),
        alpha=0.2,
    )

    final_accuracy = np.array(final_accuracy)
    return final_accuracy.mean(), final_accuracy.std(ddof=1)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    summary = []

    settings = [
        ("stationary", 6_000_000, 0),
        ("label shifts", 1_400_000, 500_000),
    ]

    for ax, (setting, rounds, drift) in zip(axes, settings):
        for model in ["NeuralBandit2", "NeuralBandit3"]:
            mean_accuracy, std_accuracy = plot_rewards(
                ax, model, rounds, drift
            )

            summary.append([
                model,
                setting,
                10,
                mean_accuracy,
                std_accuracy,
            ])

            print(
                f"{model}, {setting}: "
                f"{mean_accuracy:.2%}, std {std_accuracy:.2%}"
            )

        # mark each label shift
        if drift > 0:
            for shift in range(drift, rounds, drift):
                ax.axvline(
                    shift, color="gray", linestyle="--", alpha=0.7
                )

        ax.set_title(setting)
        ax.set_xlabel("round")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("accuracy per 10,000 rounds")
    fig.suptitle("mean and ±1 standard deviation across 10 seeds")
    fig.tight_layout()
    fig.savefig("neuralbandit_results_10seeds.png", dpi=300)
    fig.savefig("neuralbandit_results_10seeds.pdf")

    with open("neuralbandit_summary.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "model",
            "setting",
            "runs",
            "last_100000_accuracy_mean",
            "last_100000_accuracy_std",
        ])
        writer.writerows(summary)

    plt.show()


if __name__ == "__main__":
    main()