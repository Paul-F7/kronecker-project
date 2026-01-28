# Plot helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_rho_vs_M(df: pd.DataFrame, title: str = None):
    plt.figure()
    plt.plot(df["M"], df["rho"], marker="o")
    plt.xlabel("Search bound M (search k in [-M, M]^2)")
    plt.ylabel("rho_M(N, alpha)")
    plt.title(title or "rho vs search bound M")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_kstar_vs_M(df: pd.DataFrame, title: str = None):
    plt.figure()
    plt.plot(df["M"], df["k1_star"], marker="o", label="k1*")
    plt.plot(df["M"], df["k2_star"], marker="o", label="k2*")
    plt.xlabel("M")
    plt.ylabel("argmin k components")
    plt.title(title or "argmin k* vs M")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()