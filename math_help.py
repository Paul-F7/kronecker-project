## Core math 
## Math r(h) and rho_M

import numpy as np
import pandas as pd
import math

def _r_value(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """
    r(h) = max(1,|h1|) * max(1,|h2|)
    Vectorized.
    """
    return np.maximum(1.0, np.abs(h1)) * np.maximum(1.0, np.abs(h2))


def rho_box_bruteforce(N: int, alpha: float, M: int):
    """
    Slow but exact brute-force over all k1,k2 in [-M, M]^2.
    Use this to verify rho_box_numpy on small N/M.
    """
    best = float("inf")
    best_k = None
    best_h = None

    for k1 in range(-M, M + 1):
        for k2 in range(-M, M + 1):
            if k1 == 0 and k2 == 0:
                continue

            h1 = N * (k2 - alpha * k1)
            h2 = k1
            r = max(1.0, abs(h1)) * max(1.0, abs(h2))

            if r < best:
                best = r
                best_k = (k1, k2)
                best_h = (h1, h2)

    return best, best_k, best_h

import numpy as np

def rho_box1_numpy(N, alpha, M):
    """
    Fast and simple O(M) calculation using NumPy vectorization for Zaremba index
    
    Strategy:
    1. Create all k1 from -M to M.
    2. Find the nearest k2 for each k1 (plus a small window).
    3. Compute scores at once and pick the minimum.
    """

    k1 = np.arange(-M, M + 1, dtype=np.int64)
    k1 = k1[k1 != 0]

    # Find ideal k2 for k1
    ideal_k2 = alpha * k1
    nearest_k2 = np.rint(ideal_k2).astype(np.int64)
    
    # Check small window
    offsets = np.array([-1, 0, 1]) 
    
    # Create grid of k2 candidates: shape (M, 3)
    k2_candidates = nearest_k2[:, None] + offsets[None, :]
    
    # Broadcast k1 to match shape (M, 3)
    k1_grid = k1[:, None]
    
    valid_mask = np.abs(k2_candidates) <= M
    
    h1 = N * (k2_candidates - alpha * k1_grid)
    
    scores = np.maximum(1.0, np.abs(h1)) * np.abs(k1_grid)
    
    scores[~valid_mask] = np.inf
    
    min_idx_flat = np.argmin(scores)
    best_rho = scores.flat[min_idx_flat]

    row, col = np.unravel_index(min_idx_flat, scores.shape)
    best_k1 = k1_grid[row, 0]
    best_k2 = k2_candidates[row, col]
    best_h1 = h1[row, col]
    
    return float(best_rho), (int(best_k1), int(best_k2)), (float(best_h1), float(best_k1))


def rho_box_lyness(N: int, alpha: float):
    """
    Computes the exact Zaremba index using Lyness Algorithm 2. 
        From the paper A Search Program for Finding Optimal Integration Lattices*
    
    Can be used to check the others
    """
    # initialize
    k1_init = 1
    k2_init = round(alpha)
    h1_init = N * (k2_init - alpha * k1_init)
    
    # Initial upper bound (rho_u)
    best_rho = max(1.0, np.abs(h1_init)) * 1.0
    best_k = (int(k1_init), int(k2_init))
    best_h = (float(h1_init), float(k1_init))

    k1 = 1
    while k1 < best_rho:
        
        target = alpha * k1
        radius = best_rho / (N * k1)
        
        min_k2 = math.ceil(target - radius)
        max_k2 = math.floor(target + radius)
        
        # check all integers in this exact valid range
        if min_k2 <= max_k2:
            for k2 in range(min_k2, max_k2 + 1):
                # Compute exact score
                h1 = N * (k2 - target)
                score = max(1.0, abs(h1)) * k1
                
                # Update Dynamic Bound (Algorithm 3 feature)
                if score < best_rho:
                    best_rho = score
                    best_k = (int(k1), int(k2))
                    best_h = (float(h1), float(k1))
        k1 += 1
        
    return best_rho, best_k, best_h

def rho_3d_lyness(N: int, alpha: tuple):
    """
    Computes the 3D Zaremba index for an extensible sequence.
    alpha: A tuple (alpha1, alpha2) representing the 3D generator vector.
    """
    a1, a2 = alpha
    
    # Initialize with a basic k1=1 guess to establish a starting bound
    k1_init = 1
    k2_init = round(a1)
    k3_init = round(a2)
    h1_init = N * (k2_init - a1 * k1_init)
    h2_init = N * (k3_init - a2 * k1_init)
    
    # rho = k1 * max(1, |h1|) * max(1, |h2|)
    best_rho = max(1.0, abs(h1_init)) * max(1.0, abs(h2_init)) * k1_init
    best_k = (int(k1_init), int(k2_init), int(k3_init))

    k1 = 1
    # The search terminates when k1 itself exceeds the current best rho
    while k1 < best_rho:
        # For a fixed k1, we need to find k2 and k3 such that
        # k1 * |N(k2 - a1*k1)| * |N(k3 - a2*k1)| < best_rho
        
        # Max allowable 'radius' for the product of the offsets
        max_h_prod = best_rho / k1
        
        # Determine the range for k2 first
        # Since |h2| is at least 1 (per Zaremba max(1, |h|)), 
        # |h1| cannot exceed max_h_prod
        radius1 = max_h_prod / (N)
        
        target1 = a1 * k1
        min_k2 = math.ceil(target1 - radius1)
        max_k2 = math.floor(target1 + radius1)
        
        for k2 in range(min_k2, max_k2 + 1):
            h1 = N * (k2 - target1)
            h1_factor = max(1.0, abs(h1))
            
            # Now determine the remaining budget for k3
            radius2 = max_h_prod / (N * h1_factor)
            
            target2 = a2 * k1
            min_k3 = math.ceil(target2 - radius2)
            max_k3 = math.floor(target2 + radius2)
            
            for k3 in range(min_k3, max_k3 + 1):
                h2 = N * (k3 - target2)
                
                # Final 3D Score Calculation
                score = k1 * max(1.0, abs(h1)) * max(1.0, abs(h2))
                
                if score < best_rho:
                    best_rho = score
                    best_k = (k1, k2, k3)
                    # Update max_h_prod dynamically to shrink search space
                    max_h_prod = best_rho / k1 
                    
        k1 += 1
        
    return best_rho, best_k