import csv
import matplotlib.pyplot as plt
import numpy as np

def main():
    models = [
        "NeuralBandit2",
        "NeuralBandit3",
        "LinUCB",
        "CTS",
        "Banditron",
    ]

    settings = [("stationary", 2_500_000, 0),("label shifts", 2_500_000, 500_000),]


    with np.load("reference_predictions.npz") as data:
        predictions = data["predictions"]
        labels = data["labels"]

    reference_rewards = (predictions == labels).astype(np.int64)
    row_order = np.random.default_rng(67).permutation(len(labels))

    block_size = 10_000
    summary = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, (setting, rounds, drift) in zip(axes, settings):
        reference_blocks = []

        for seed in range(67, 77):
            rng = np.random.default_rng(seed)
            start = rng.integers(len(row_order))
            positions = (start + np.arange(rounds)) % len(row_order)
            rows = row_order[positions]

            rewards = reference_rewards[rows]
            reference_blocks.append(
                rewards.reshape(-1, block_size).sum(axis=1)
            )

        for model in models:
            curves = []

            for run, seed in enumerate(range(67, 77)):
                filename = (
                    f"{model}_rounds{rounds}"
                    f"_drift{drift}_seed{seed}.npz"
                )

                with np.load(filename) as data:
                    rewards = data["rewards"]

                if rewards.shape != (rounds,):
                    raise ValueError(f"incomplete run in {filename}")

                rewards = rewards.astype(np.int64)
                bandit_blocks = rewards.reshape(-1, block_size).sum(axis=1)

                difference = reference_blocks[run] - bandit_blocks
                regret = np.cumsum(difference)
                curves.append(np.concatenate(([0], regret)))

            curves = np.array(curves)
            mean_regret = curves.mean(axis=0)
            std_regret = curves.std(axis=0, ddof=1)
            round_numbers = np.arange(0, rounds + 1, block_size)

            line, = ax.plot(round_numbers, mean_regret, label=model)
            ax.fill_between(
                round_numbers,
                mean_regret - std_regret,
                mean_regret + std_regret,
                color=line.get_color(),
                alpha=0.15,
            )

            final_mean = mean_regret[-1]
            final_std = std_regret[-1]
            summary.append([model, setting, 10, final_mean, final_std])

            print(
                f"{model}, {setting}: "
                f"regret {final_mean:,.0f}, std {final_std:,.0f}"
            )

        if drift > 0:
            for shift in range(drift, rounds, drift):
                ax.axvline(
                    shift, color="gray", linestyle="--", alpha=0.7
                )

        ax.set_title(setting)
        ax.set_xlabel("round")
        ax.set_ylabel("cumulative reward difference from reference MLP")
        ax.legend()
        ax.grid(alpha=0.2)

    fig.suptitle("mean and ±1 standard deviation across 10 seeds")
    fig.tight_layout()
    fig.savefig("reference_regret.png", dpi=300)
    fig.savefig("reference_regret.pdf")

    with open("reference_regret_summary.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "model", "setting", "runs",
            "final_regret_mean", "final_regret_std",
        ])
        writer.writerows(summary)

    plt.show()

if __name__ == "__main__":
    main()