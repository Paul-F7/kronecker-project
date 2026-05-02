"""Generate static PNGs for simple-site/index.html."""
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(ROOT)
sys.path.insert(0, DATA)
from math_help import rho_box_lyness

rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#222",
    "xtick.color": "#444",
    "ytick.color": "#444",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "#fdfdfa",
    "axes.facecolor": "#fdfdfa",
    "figure.facecolor": "#fdfdfa",
})

PHI = (1 + 5 ** 0.5) / 2
ALPHA_GOLD = PHI - 1                # 0.6180...
ALPHA_SQRT2 = 2 ** 0.5 - 1          # 0.4142...
ALPHA_E = math.e - 2                # 0.7183...
ALPHA_PI = math.pi - 3              # 0.1416...


def kronecker_2d(n, a1, a2):
    i = np.arange(1, n + 1)
    return np.column_stack([(i * a1) % 1, (i * a2) % 1])


# ---------- Plot 1: Kronecker vs Random scatter ----------
def plot_scatter():
    n = 300
    rng = np.random.default_rng(7)
    pts_k = kronecker_2d(n, ALPHA_GOLD, ALPHA_SQRT2)
    pts_r = rng.random((n, 2))

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.1))
    for ax, pts, title in (
        (axes[0], pts_k, f"Kronecker ($\\alpha = \\varphi-1,\\ \\sqrt{{2}}-1$)"),
        (axes[1], pts_r, "Uniform random"),
    ):
        ax.scatter(pts[:, 0], pts[:, 1], s=8, color="#1f3b66", alpha=0.85, edgecolors="none")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.15)
    fig.suptitle(f"{n} points in the unit square", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "plot_scatter.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------- Plot 2: rho vs N for 4 famous alphas ----------
def plot_rho_vs_n():
    Ns = np.arange(2, 401)
    series = [
        ("golden ratio  $\\varphi-1$", ALPHA_GOLD, "#1f3b66"),
        ("$\\sqrt{2}-1$",              ALPHA_SQRT2, "#3f7d3a"),
        ("$e-2$",                      ALPHA_E,    "#b85c00"),
        ("$\\pi-3$",                   ALPHA_PI,   "#a8323a"),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for label, a, color in series:
        rhos = []
        for N in Ns:
            r, _, _ = rho_box_lyness(int(N), float(a))
            rhos.append(r)
        ax.plot(Ns, rhos, lw=1.2, color=color, label=label)
    ax.set_xlabel("N (number of points)")
    ax.set_ylabel(r"Zaremba index $\rho$")
    ax.set_title(r"$\rho(\alpha, N)$ for four irrationals — clean lines stay clean")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "plot_rho_vs_n.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------- Plot 3: Distribution of rho at fixed N (4D, N=1009) ----------
def plot_distribution():
    npz_path = os.path.join(DATA, "zaremba_4d_N1009.npz")
    d = np.load(npz_path)
    rhos = d["rhos"].astype(np.int64)
    weights = d["weights"].astype(np.int64)

    total = int(weights.sum())
    # Weighted percentiles via cumulative sum on sorted rhos
    order = np.argsort(rhos)
    rs = rhos[order]; ws = weights[order]
    cs = np.cumsum(ws)
    def pct(p):
        idx = np.searchsorted(cs, p * total)
        return int(rs[min(idx, len(rs) - 1)])
    p50 = pct(0.50); p90 = pct(0.90); p99 = pct(0.999); pmax = int(rs[-1])

    N = 1009
    rho_max = pmax
    bins = np.linspace(0, rho_max, 80)
    counts, edges = np.histogram(rhos, bins=bins, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.95,
           color="#4a6fa5", edgecolor="none", alpha=0.85)
    ax.axvline(p50, color="#a8323a", lw=1.2, ls="--", label=f"median  ρ = {p50}")
    ax.axvline(p99, color="#b85c00", lw=1.2, ls="--", label=f"99.9th percentile  ρ = {p99}")
    ax.axvline(pmax, color="#3f7d3a", lw=1.2, ls="--", label=f"best found  ρ = {pmax}")
    ax.set_xlabel(r"Zaremba index $\rho$")
    ax.set_ylabel("number of generators (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"4D generator quality at N = {N}  ({total:,} generators)")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "plot_distribution.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_scatter()
    print("scatter done")
    plot_rho_vs_n()
    print("rho vs N done")
    plot_distribution()
    print("distribution done")
