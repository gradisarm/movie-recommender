"""Figure generation for the movie-recommender experiments.

Kept separate from main.py so the modeling/evaluation logic stays readable and
all matplotlib code lives in one place. Each function saves one report figure
to FIG_DIR. Uses the non-interactive Agg backend (no display needed).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

FIG_DIR = "figures"


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_rmse_comparison(rmses):
    """Experiment 1 (primary): RMSE of SVD vs mean-based baselines. Lower is better.

    rmses: dict {label: rmse}, insertion order = bar order (SVD last/highlighted).
    """
    label_sl = {"global-mean": "globalno povprečje",
                "user-mean": "povprečje uporabnika",
                "item-mean": "povprečje filma"}
    labels = [label_sl.get(k, k) for k in rmses.keys()]
    values = list(rmses.values())
    colors = ["#999999"] * (len(labels) - 1) + ["#1f77b4"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("RMSE (nižje je bolje)")
    ax.set_title("Točnost napovedovanja ocen: SVD proti izhodiščnim modelom")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return _save(fig, "rmse_comparison.png")


def plot_precision_comparison(pop_p, svd_p, n, default_k, k_at):
    """Honest secondary finding: popularity vs SVD on precision@k (popularity wins)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Priljubljenost\n(izhodišče)", f"SVD (k={default_k})"], [pop_p, svd_p],
           color=["#999999", "#1f77b4"])
    ax.set_ylabel(f"Preciznost@{k_at}")
    ax.set_title(f"Razvrščanje top-N: priljubljenost je močno izhodišče (n={n})")
    for i, v in enumerate([pop_p, svd_p]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    return _save(fig, "precision_comparison.png")


def plot_ksweep(results):
    """Experiment 2: RMSE vs k. results = list of (k, rmse)."""
    ks, vals = zip(*results)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, vals, "o-", color="#1f77b4")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("k (število latentnih faktorjev)")
    ax.set_ylabel("RMSE (nižje je bolje)")
    ax.set_title("Pregled k: kompromis med premajhnim in prekomernim prileganjem")
    return _save(fig, "k_sweep.png")


def plot_learning_curve(results, n_eval_users):
    """Experiment 3: RMSE vs training size. results = list of (n_users, rmse)."""
    xs, vals = zip(*results)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, vals, "o-", color="#2ca02c")
    ax.set_xlabel("Število uporabnikov v učni množici")
    ax.set_ylabel(f"RMSE na fiksni testni množici ({n_eval_users} uporabnikov, nižje je bolje)")
    ax.set_title("Učna krivulja: več podatkov → boljše napovedi")
    return _save(fig, "learning_curve.png")
