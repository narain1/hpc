#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NX 1000         // Number of grid points in the x direction
#define NY 1000         // Number of grid points in the y direction
#define NSTEPS 1000    // Number of time steps
#define ALPHA 0.01     // Thermal diffusivity
#define DX 0.01        // Grid spacing in the x direction
#define DY 0.01        // Grid spacing in the y direction
#define DT 0.0001      // Time step

// Boundary conditions
void apply_boundary_conditions(double T[NX][NY]) {
    for (int i = 0; i < NX; i++) {
        T[i][0] = 100.0;          // Top boundary (constant high temperature)
        T[i][NY-1] = 0.0;         // Bottom boundary (constant low temperature)
    }
    for (int j = 0; j < NY; j++) {
        T[0][j] = 100.0;          // Left boundary (constant high temperature)
        T[NX-1][j] = 0.0;         // Right boundary (constant low temperature)
    }
}

// Parallelized heat transfer update function
void update_temperature(double T[NX][NY], double T_new[NX][NY]) {
    #pragma omp parallel for collapse(2) // OpenMP parallel loop
    for (int i = 1; i < NX-1; i++) {
        for (int j = 1; j < NY-1; j++) {
            T_new[i][j] = T[i][j] + ALPHA * DT * (
                (T[i+1][j] - 2.0 * T[i][j] + T[i-1][j]) / (DX * DX) +
                (T[i][j+1] - 2.0 * T[i][j] + T[i][j-1]) / (DY * DY)
            );
        }
    }
}

int main() {
    double T[NX][NY] = {0};     // Temperature array
    double T_new[NX][NY] = {0}; // New temperature array after each step

    // Apply boundary conditions
    apply_boundary_conditions(T);

    // Time-stepping loop
    for (int step = 0; step < NSTEPS; step++) {
        // Update temperature in the interior of the grid
        update_temperature(T, T_new);

        // Swap arrays (T_new becomes T for the next step)
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                T[i][j] = T_new[i][j];
            }
        }

        // Re-apply boundary conditions
        apply_boundary_conditions(T);

        // (Optional) Print out intermediate temperature results
        if (step % 100 == 0) {
            printf("Step %d complete\n", step);
        }
    }

    // Output the final temperature distribution
    FILE *fout = fopen("temperature_distribution.dat", "w");
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            fprintf(fout, "%lf ", T[i][j]);
        }
        fprintf(fout, "\n");
    }
    fclose(fout);

    printf("Simulation complete. Final temperature distribution written to 'temperature_distribution.dat'\n");
    return 0;
}
