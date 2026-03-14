## Core math 
## Math r(h) and rho_M

import numpy as np
import pandas as pd
import math
from collections import Counter

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



def generate_optimized_g_vectors_3d(N):
    """
    Generates symmetry-optimized g-vectors for 3 dimensions modulo N.
    Enforces g2 <= g3 to cut the search space in half.
    
    Yields:
        vector (tuple): The (1, g2, g3) vector to test.
        weight (int): 1 if the vector is symmetric (g2 == g3).
                      2 if asymmetric (represents both (1, g2, g3) and (1, g3, g2)).
    """
    # In the literature, g vectors usually range from 1 to N-1
    for g2 in range(1, N):
        for g3 in range(g2, N): # Start g3 at g2 to enforce g2 <= g3
            
            # If they are equal, this represents 1 unique vector in the full space
            if g2 == g3:
                weight = 1
            # If they are different, it represents 2 vectors (g2, g3) and (g3, g2)
            else:
                weight = 2
                
            yield (1, g2, g3), weight

# ==========================================
# Example usage and mathematical verification
# ==========================================
if __name__ == "__main__":
    N_test = 5 # Small prime for testing
    
    total_original_space = (N_test - 1) ** 2
    optimized_count = 0
    reconstructed_count = 0
    
    print(f"Testing N = {N_test}")
    print(f"Theoretical full search space: {total_original_space} vectors\n")
    print("Optimized Vectors to Compute:")
    print("-----------------------------")
    
    for g_vector, weight in generate_optimized_g_vectors_3d(N_test):
        print(f"Vector: {g_vector} | Weight: {weight}")
        optimized_count += 1
        reconstructed_count += weight
        
    print("-----------------------------")
    print(f"Actual $\\rho$ calculations performed: {optimized_count}")
    print(f"Total vector equivalents checked (sum of weights): {reconstructed_count}")
    
    assert reconstructed_count == total_original_space, "Weight math is incorrect!"



def rho_4d_lyness(N: int, alpha: tuple):
    """
    Computes the 4D Zaremba index.
    alpha: A tuple (a1, a2, a3) representing the continuous generator vector.
    """
    a1, a2, a3 = alpha
    
    k1_init = 1
    k2_init = round(a1)
    k3_init = round(a2)
    k4_init = round(a3)
    
    h1_init = N * (k2_init - a1 * k1_init)
    h2_init = N * (k3_init - a2 * k1_init)
    h3_init = N * (k4_init - a3 * k1_init)
    
    best_rho = max(1.0, abs(h1_init)) * max(1.0, abs(h2_init)) * max(1.0, abs(h3_init)) * k1_init
    best_k = (int(k1_init), int(k2_init), int(k3_init), int(k4_init))

    k1 = 1
    while k1 < best_rho:
        max_h_prod = best_rho / k1
        
        # Determine range for k2
        radius1 = max_h_prod / N
        target1 = a1 * k1
        min_k2 = math.ceil(target1 - radius1)
        max_k2 = math.floor(target1 + radius1)
        
        for k2 in range(min_k2, max_k2 + 1):
            h1 = N * (k2 - target1)
            h1_factor = max(1.0, abs(h1))
            
            # Determine range for k3
            radius2 = max_h_prod / (N * h1_factor)
            target2 = a2 * k1
            min_k3 = math.ceil(target2 - radius2)
            max_k3 = math.floor(target2 + radius2)
            
            for k3 in range(min_k3, max_k3 + 1):
                h2 = N * (k3 - target2)
                h2_factor = max(1.0, abs(h2))
                
                # Determine range for k4
                radius3 = max_h_prod / (N * h1_factor * h2_factor)
                target3 = a3 * k1
                min_k4 = math.ceil(target3 - radius3)
                max_k4 = math.floor(target3 + radius3)
                
                for k4 in range(min_k4, max_k4 + 1):
                    h3 = N * (k4 - target3)
                    score = k1 * h1_factor * h2_factor * max(1.0, abs(h3))
                    
                    if score < best_rho:
                        best_rho = score
                        best_k = (k1, k2, k3, k4)
                        max_h_prod = best_rho / k1 
                        
        k1 += 1
        
    return best_rho, best_k


def rho_5d_lyness(N: int, alpha: tuple):
    """
    Computes the 5D Zaremba index.
    alpha: A tuple (a1, a2, a3, a4) representing the continuous generator vector.
    """
    a1, a2, a3, a4 = alpha
    
    k1_init = 1
    k2_init = round(a1)
    k3_init = round(a2)
    k4_init = round(a3)
    k5_init = round(a4)
    
    h1_init = N * (k2_init - a1 * k1_init)
    h2_init = N * (k3_init - a2 * k1_init)
    h3_init = N * (k4_init - a3 * k1_init)
    h4_init = N * (k5_init - a4 * k1_init)
    
    best_rho = max(1.0, abs(h1_init)) * max(1.0, abs(h2_init)) * max(1.0, abs(h3_init)) * max(1.0, abs(h4_init)) * k1_init
    best_k = (int(k1_init), int(k2_init), int(k3_init), int(k4_init), int(k5_init))

    k1 = 1
    while k1 < best_rho:
        max_h_prod = best_rho / k1
        
        radius1 = max_h_prod / N
        target1 = a1 * k1
        min_k2 = math.ceil(target1 - radius1)
        max_k2 = math.floor(target1 + radius1)
        
        for k2 in range(min_k2, max_k2 + 1):
            h1 = N * (k2 - target1)
            h1_factor = max(1.0, abs(h1))
            
            radius2 = max_h_prod / (N * h1_factor)
            target2 = a2 * k1
            min_k3 = math.ceil(target2 - radius2)
            max_k3 = math.floor(target2 + radius2)
            
            for k3 in range(min_k3, max_k3 + 1):
                h2 = N * (k3 - target2)
                h2_factor = max(1.0, abs(h2))
                
                radius3 = max_h_prod / (N * h1_factor * h2_factor)
                target3 = a3 * k1
                min_k4 = math.ceil(target3 - radius3)
                max_k4 = math.floor(target3 + radius3)
                
                for k4 in range(min_k4, max_k4 + 1):
                    h3 = N * (k4 - target3)
                    h3_factor = max(1.0, abs(h3))
                    
                    radius4 = max_h_prod / (N * h1_factor * h2_factor * h3_factor)
                    target4 = a4 * k1
                    min_k5 = math.ceil(target4 - radius4)
                    max_k5 = math.floor(target4 + radius4)
                    
                    for k5 in range(min_k5, max_k5 + 1):
                        h4 = N * (k5 - target4)
                        score = k1 * h1_factor * h2_factor * h3_factor * max(1.0, abs(h4))
                        
                        if score < best_rho:
                            best_rho = score
                            best_k = (k1, k2, k3, k4, k5)
                            max_h_prod = best_rho / k1 
                            
        k1 += 1
        
    return best_rho, best_k


def generate_optimized_g_vectors_4d(N):
    """
    Generates symmetry-optimized and composite-safe vectors for 4D.
    Enforces g2 <= g3 <= g4.
    """
    for g2 in range(1, N):
        if math.gcd(g2, N) != 1: 
            continue
            
        for g3 in range(g2, N):
            if math.gcd(g3, N) != 1: 
                continue
                
            for g4 in range(g3, N):
                if math.gcd(g4, N) != 1: 
                    continue
                
                # Calculate permutation weight: 3! / (duplicates!)
                counts = Counter([g2, g3, g4]).values()
                weight = 6  # 3!
                for count in counts:
                    weight //= math.factorial(count)
                    
                yield (1, g2, g3, g4), weight


def generate_optimized_g_vectors_5d(N):
    """
    Generates symmetry-optimized and composite-safe vectors for 5D.
    Enforces g2 <= g3 <= g4 <= g5.
    """
    for g2 in range(1, N):
        if math.gcd(g2, N) != 1: 
            continue
            
        for g3 in range(g2, N):
            if math.gcd(g3, N) != 1: 
                continue
                
            for g4 in range(g3, N):
                if math.gcd(g4, N) != 1: 
                    continue
                    
                for g5 in range(g4, N):
                    if math.gcd(g5, N) != 1: 
                        continue
                    
                    # Calculate permutation weight: 4! / (duplicates!)
                    counts = Counter([g2, g3, g4, g5]).values()
                    weight = 24  # 4!
                    for count in counts:
                        weight //= math.factorial(count)
                        
                    yield (1, g2, g3, g4, g5), weight






