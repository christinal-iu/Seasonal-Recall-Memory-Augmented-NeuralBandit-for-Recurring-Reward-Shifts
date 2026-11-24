import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t


seeds = [90, 91, 92, 93, 94]

names = {
    "original": "original NB3",
    "explore": "NB3, 5% exploration",
    "memory": "seasonal recall",
    "fifo": "seasonal recall, FIFO",
}

colors = {
    "original": "tab:blue",
    "explore": "tab:orange",
    "memory": "tab:green",
    "fifo": "tab:purple",
}


def load_rewards(dataset, method, setting):
    rewards = []

    for seed in seeds:
        filename = (
            f"final_results/final_v1_{dataset}"
            f"_{method}_{setting}"
            f"_rounds100000_interval25000"
            f"_seed{seed}.npz"
        )

        with np.load(filename) as data:
            values = data["rewards"].copy()

        if values.shape != (100000,):
            raise ValueError(f"unexpected length: {filename}")

        rewards.append(values)

    return np.array(rewards)


def plot_learning_curves():
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8),
        sharex=True,
        sharey=True,
    )

    window = 1000

    for row, dataset in enumerate(["covertype", "digits"]):
        for col, setting in enumerate([
            "stationary",
            "recurring",
            "novel",
        ]):
            ax = axes[row, col]
            methods = ["original", "explore", "memory"]

            if setting == "recurring":
                methods.append("fifo")

            for method in methods:
                rewards = load_rewards(
                    dataset, method, setting
                )

                # average each nonoverlapping block of 1000 rounds
                blocks = rewards.reshape(
                    len(seeds), -1, window
                ).mean(axis=2)

                mean = blocks.mean(axis=0)
                std = blocks.std(axis=0, ddof=1)

                x = (
                    np.arange(len(mean)) * window
                    + window / 2
                )

                ax.plot(
                    x,
                    mean,
                    label=names[method],
                    color=colors[method],
                    linewidth=1.5,
                )

                ax.fill_between(
                    x,
                    np.clip(mean - std, 0, 1),
                    np.clip(mean + std, 0, 1),
                    color=colors[method],
                    alpha=0.12,
                )

            if setting != "stationary":
                for boundary in [25000, 50000, 75000]:
                    ax.axvline(
                        boundary,
                        color="gray",
                        linestyle="--",
                        linewidth=1,
                    )

            ax.set_title(f"{dataset}: {setting}")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
            ax.ticklabel_format(
                axis="x", style="sci", scilimits=(0, 0)
            )

            if row == 1:
                ax.set_xlabel("round")

            if col == 0:
                ax.set_ylabel("accuracy per 1,000 rounds")

    handles, labels = axes[0, 1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
    )

    fig.suptitle(
        "mean and ±1 standard deviation across five seeds"
    )

    fig.tight_layout(rect=[0, 0.07, 1, 0.95])

    fig.savefig(
        "figures/final_learning_curves.png",
        dpi=300,
    )
    fig.savefig(
        "figures/final_learning_curves.pdf"
    )

    plt.close(fig)


def recurring_metrics(rewards):
    overall = rewards.mean(axis=1)
    returning = rewards[:, 50000:].mean(axis=1)

    early = np.concatenate([
        rewards[:, 50000:51000],
        rewards[:, 75000:76000],
    ], axis=1).mean(axis=1)

    return np.column_stack([
        overall,
        returning,
        early,
    ])


def plot_paired_gains():
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True
    )

    comparisons = [
        ("explore", "recall minus exploration baseline"),
        ("fifo", "recall minus FIFO archive"),
    ]

    x = np.arange(3)
    width = 0.34

    for ax, (baseline, title) in zip(axes, comparisons):
        for index, dataset in enumerate([
            "covertype",
            "digits",
        ]):
            memory = recurring_metrics(
                load_rewards(dataset, "memory", "recurring")
            )

            control = recurring_metrics(
                load_rewards(dataset, baseline, "recurring")
            )

            # compare methods within matching seeds
            differences = 100 * (memory - control)

            mean = differences.mean(axis=0)
            standard_error = (
                differences.std(axis=0, ddof=1)
                / np.sqrt(len(seeds))
            )

            margin = (
                t.ppf(0.975, len(seeds) - 1)
                * standard_error
            )

            offset = (index - 0.5) * width

            ax.bar(
                x + offset,
                mean,
                width=width,
                yerr=margin,
                capsize=4,
                label=dataset,
                color=["tab:green", "tab:orange"][index],
                alpha=0.85,
            )

        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([
            "overall",
            "returning\npatterns",
            "first 1,000 rounds\nafter returns",
        ])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("accuracy difference, percentage points")
    axes[1].legend()

    fig.suptitle(
        "recurring shifts: paired gains with approximate 95% CIs"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig.savefig(
        "figures/final_paired_gains.png",
        dpi=300,
    )
    fig.savefig(
        "figures/final_paired_gains.pdf"
    )

    plt.close(fig)


def main():
    os.makedirs("figures", exist_ok=True)

    plot_learning_curves()
    plot_paired_gains()

    print("saved learning curves as PNG and PDF")
    print("saved paired gains as PNG and PDF")


if __name__ == "__main__":
    main()