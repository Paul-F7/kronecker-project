import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot import plot_kstar_vs_M, plot_rho_vs_M
from experiment import bound_sufficiency_df, is_N_enough
import math
'''
# Checks
N = 200
alpha = (math.sqrt(5) - 1) / 2  # golden ratio conjugate ~0.618

df = bound_sufficiency_df(N, alpha, multipliers=(1, 2, 4, 8), k2_window=2, method="numpy")
print(df)
print("Is M=N enough (based on mult=1 vs 2)?", is_N_enough(df))

plot_rho_vs_M(df, title=f"rho vs M (N={N}, alpha≈{alpha:.6f})")
plot_kstar_vs_M(df, title=f"k* vs M (N={N}, alpha≈{alpha:.6f})")
'''

N = 30
alpha = (math.sqrt(5)-1)/2

df_fast = bound_sufficiency_df(N, alpha, multipliers=(1,2), method="numpy", k2_window=3)
df_brut = bound_sufficiency_df(N, alpha, multipliers=(1,2), method="bruteforce")
df_lyness = bound_sufficiency_df(N, alpha, multipliers=(1,2), method="lyness")

df_fast, df_brut

print("Fast method:")
print(df_fast)
print("\nBrute force method:")
print(df_brut)
print("\nLyness force method:")
print(df_lyness)

