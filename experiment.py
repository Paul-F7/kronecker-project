# Experiment,  Is  M=N enough?

from math_help import _r_value, rho_box_numpy, rho_box_bruteforce, rho_box1_numpy, rho_box_lyness
import numpy as np
import pandas as pd


def bound_sufficiency_df(
    N: int,
    alpha: float,
    multipliers=(1/2, 1, 2, 4, 8),
    k2_window: int = 2,
    method: str = "numpy",
):
    """
    Compute rho_M for M = mult*N and return a DataFrame.

    method:
      - "numpy": fast, uses rho_box_numpy
      - "bruteforce": exact, slow, uses rho_box_bruteforce (only for small N)
    """
    rows = []

    for mult in multipliers:
        M = int(mult * N)

        if method == "numpy":
            rho, k_star, h_star = rho_box1_numpy(N, alpha, M)
        elif method == "bruteforce":
            rho, k_star, h_star = rho_box_bruteforce(N, alpha, M)
        else:
            raise ValueError("method must be 'numpy' or 'bruteforce'.")

        rows.append({
            "N": N,
            "alpha": alpha,
            "mult": mult,
            "M": M,
            "rho": rho,
            "k1_star": k_star[0],
            "k2_star": k_star[1],
            "h1_star": h_star[0],
            "h2_star": h_star[1],
        })

    return pd.DataFrame(rows)


def is_N_enough(df: pd.DataFrame, tol: float = 1e-9) -> bool:
    """
    Decide if M=N seems enough by comparing results at mult=1 and mult=2.
    You can strengthen this by also checking mult=4,8.
    """
    d1 = df[df["mult"] == 1].iloc[0]
    d2 = df[df["mult"] == 2].iloc[0]

    same_rho = np.abs(d1["rho"] - d2["rho"]) <= tol * np.maximum(1.0, np.abs(d2["rho"]))
    same_k = (d1["k1_star"] == d2["k1_star"]) and (d1["k2_star"] == d2["k2_star"])
    return bool(same_rho and same_k)