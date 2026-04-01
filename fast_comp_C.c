#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Fast inline math helpers
static inline int int_abs(int x) { return x < 0 ? -x : x; }
static inline int int_max(int a, int b) { return a > b ? a : b; }

// Greatest Common Divisor
int gcd(int a, int b) {
    while (b) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// 5D Zaremba Branch and Bound
int rho_5d_c(int N, int g2, int g3, int g4, int g5) {
    // 1. Aggressive Early Exit for rho = 1
    // In 5D, a massive percentage of vectors have rho=1. We check this instantly.
    for (int t2 = -1; t2 <= 1; t2++) {
        for (int t3 = -1; t3 <= 1; t3++) {
            for (int t4 = -1; t4 <= 1; t4++) {
                for (int t5 = -1; t5 <= 1; t5++) {
                    if (t2 == 0 && t3 == 0 && t4 == 0 && t5 == 0) continue;
                    
                    int s = t2*g2 + t3*g3 + t4*g4 + t5*g5;
                    int h1 = (-s) % N;
                    if (h1 < 0) h1 += N;
                    if (h1 > N / 2) h1 -= N;
                    
                    if (int_abs(h1) <= 1) {
                        return 1; // Instant exit
                    }
                }
            }
        }
    }

    int best_rho = N;

    // 2. Singleton Bounds to lower best_rho immediately
    int gs[4] = {g2, g3, g4, g5};
    for(int i = 0; i < 4; i++) {
        for(int sign = -1; sign <= 1; sign += 2) {
            int h1 = (-(sign * gs[i])) % N;
            if (h1 < 0) h1 += N;
            if (h1 > N / 2) h1 -= N;
            int r = int_max(1, int_abs(h1));
            if (r < best_rho) best_rho = r;
        }
    }

    // 3. Deep Branch and Bound
    for (int h2 = 0; h2 < best_rho; h2++) {
        int f2 = int_max(1, int_abs(h2));
        if (f2 >= best_rho) break;
        
        int limit3 = (best_rho - 1) / f2;
        int start3 = (h2 == 0) ? 0 : -limit3;
        
        for (int h3 = start3; h3 <= limit3; h3++) {
            int f3 = int_max(1, int_abs(h3));
            int p23 = f2 * f3;
            if (p23 >= best_rho) continue;
            
            int limit4 = (best_rho - 1) / p23;
            int start4 = (h2 == 0 && h3 == 0) ? 0 : -limit4;
            
            for (int h4 = start4; h4 <= limit4; h4++) {
                int f4 = int_max(1, int_abs(h4));
                int p234 = p23 * f4;
                if (p234 >= best_rho) continue;
                
                int limit5 = (best_rho - 1) / p234;
                int start5 = (h2 == 0 && h3 == 0 && h4 == 0) ? 1 : -limit5;
                
                for (int h5 = start5; h5 <= limit5; h5++) {
                    int f5 = int_max(1, int_abs(h5));
                    int p2345 = p234 * f5;
                    if (p2345 >= best_rho) continue;
                    
                    // Core Dual Lattice Condition
                    int h1 = (-(h2*g2 + h3*g3 + h4*g4 + h5*g5)) % N;
                    if (h1 < 0) h1 += N;
                    if (h1 > N / 2) h1 -= N;
                    
                    int score = p2345 * int_max(1, int_abs(h1));
                    if (score < best_rho) {
                        best_rho = score;
                    }
                }
            }
        }
    }
    return best_rho;
}

int main(int argc, char *argv[]) {
    // START SMALL! O(N^4) grows incredibly fast. 
    // Test with N=101 first, then move to 251, 401, etc.
    if (argc < 2) {
        printf("Error: You must provide an N value.\n");
        printf("Usage: ./zaremba_5d <N>\n");
        return 1;
    }
    int N = atoi(argv[1]);

    char filename[50];
    sprintf(filename, "zaremba_5d_N%d.csv", N);
    
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    fprintf(f, "rho\n"); // CSV Header

    printf("Starting 5D Calculation for N = %d...\n", N);
    double start_time = omp_get_wtime();
    
    int processed_vectors = 0;

    // Parallelize the outermost loop across all CPU cores
    #pragma omp parallel for schedule(dynamic) reduction(+:processed_vectors)
    for (int g2 = 1; g2 < N; g2++) {
        if (gcd(g2, N) != 1) continue;
        
        for (int g3 = g2; g3 < N; g3++) {
            if (gcd(g3, N) != 1) continue;
            
            for (int g4 = g3; g4 < N; g4++) {
                if (gcd(g4, N) != 1) continue;
                
                for (int g5 = g4; g5 < N; g5++) {
                    if (gcd(g5, N) != 1) continue;
                    
                    int rho = rho_5d_c(N, g2, g3, g4, g5);
                    processed_vectors++;
                    
                    // Thread-safe file writing
                    #pragma omp critical
                    {
                        fprintf(f, "%d\n", rho);
                    }
                }
            }
        }
    }
    
    fclose(f);
    double end_time = omp_get_wtime();
    
    printf("Finished! Processed %d valid vectors.\n", processed_vectors);
    printf("Time elapsed: %.2f seconds.\n", end_time - start_time);
    printf("Results saved to %s\n", filename);
    
    return 0;
}