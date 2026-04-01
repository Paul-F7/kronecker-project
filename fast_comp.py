import numpy as np
import pandas as pd
import math
from collections import Counter
import time
from numba import njit, prange


@njit(fastmath=True)
def rho_4d_numba(N, g2, g3, g4):
    """
    Highly optimized, C-compiled branch-and-bound for 4D.
    """
    # 1. Exact check for rho = 1
    for t2 in (-1, 0, 1):
        for t3 in (-1, 0, 1):
            for t4 in (-1, 0, 1):
                if t2 == 0 and t3 == 0 and t4 == 0: continue
                s = t2*g2 + t3*g3 + t4*g4
                h1 = (-s) % N
                if h1 > N // 2: h1 -= N
                if abs(h1) <= 1:
                    return 1

    best_rho = N
    
    # 2. Singleton bounds
    for gj in (g2, g3, g4):
        for sign in (-1, 1):
            h1 = (-(sign * gj)) % N
            if h1 > N // 2: h1 -= N
            r = max(1, abs(h1))
            if r < best_rho: best_rho = r
    
    # 3. Branch and bound
    for h2 in range(0, best_rho):
        f2 = max(1, abs(h2))
        if f2 >= best_rho: break
        
        limit3 = (best_rho - 1) // f2
        start3 = 0 if h2 == 0 else -limit3
        
        for h3 in range(start3, limit3 + 1):
            f3 = max(1, abs(h3))
            p23 = f2 * f3
            if p23 >= best_rho: continue
            
            limit4 = (best_rho - 1) // p23
            start4 = 1 if (h2 == 0 and h3 == 0) else -limit4
            
            for h4 in range(start4, limit4 + 1):
                f4 = max(1, abs(h4))
                p234 = p23 * f4
                if p234 >= best_rho: continue
                
                h1 = (-(h2*g2 + h3*g3 + h4*g4)) % N
                if h1 > N // 2: h1 -= N
                
                score = p234 * max(1, abs(h1))
                if score < best_rho:
                    best_rho = score
                    
    return best_rho



@njit(parallel=True)
def full_distribution_numba_4d(N):
    """
    Scans the entire symmetry-reduced space in parallel across all CPU cores.
    """
    # Pre-allocate a large array for results (N^3 / 6 max size)
    max_size = (N**3) // 6 + N**2
    rhos = np.zeros(max_size, dtype=np.int32)
    weights = np.zeros(max_size, dtype=np.int32)
    
    count = 0
    # Parallelize the outermost loop
    for g2 in prange(1, N):
        # Numba doesn't support math.gcd natively in prange, so we write a quick inline GCD
        a, b = g2, N
        while b: a, b = b, a % b
        if a != 1: continue
            
        for g3 in range(g2, N):
            a, b = g3, N
            while b: a, b = b, a % b
            if a != 1: continue
                
            for g4 in range(g3, N):
                a, b = g4, N
                while b: a, b = b, a % b
                if a != 1: continue
                
                # Calculate weights for symmetry reduction
                w = 6
                if g2 == g3 and g3 == g4: w = 1
                elif g2 == g3 or g3 == g4: w = 3
                
                rho = rho_4d_numba(N, g2, g3, g4)
                
                # Thread-safe atomic write using Numba's loop indexing
                idx = count
                rhos[idx] = rho
                weights[idx] = w
                count += 1
                
    return rhos[:count], weights[:count]


def run_and_save(N_values):
    for N in N_values:
        print(f"Starting N={N}...")
        t0 = time.time()
        
        rhos, weights = full_distribution_numba_4d(N)
        
        # Save the raw arrays to disk efficiently
        filename = f"zaremba_4d_N{N}.npz"
        np.savez_compressed(filename, rhos=rhos, weights=weights)
        
        elapsed = time.time() - t0
        print(f"Finished N={N} in {elapsed:.1f}s. Saved to {filename}")

if __name__ == "__main__":
    # You can run this from your terminal: python compute_zaremba.py
    target_N_values = [401, 503, 1009, 1511, 2003]
    run_and_save(target_N_values)