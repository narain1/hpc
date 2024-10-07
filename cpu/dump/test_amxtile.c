#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define MATRIX_SIZE 4096
#define TILE_DIM 16

// Define and align the tile configuration structure to 64 bytes
__attribute__((aligned(64))) static const uint8_t tileconfig[64] = {
    16, 0, TILE_DIM, 0,  // Tile 0 configuration (used for accumulation)
    16, 0, TILE_DIM, 0,  // Tile 1 configuration (used for A)
    16, 0, TILE_DIM, 0,  // Tile 2 configuration (used for B)
    0, 0, 0, 0, 0, 0, 0, 0,  // Padding
    0, 0, 0, 0, 0, 0, 0, 0,  // Padding
    0, 0, 0, 0, 0, 0, 0, 0,  // Padding
    0, 0, 0, 0, 0, 0, 0, 0   // Padding
};

void amx_tile_mm(float* A, float* B, float* C, int n) {
    // Initialize AMX with the tile configuration
    _tile_loadconfig((const void *)tileconfig);
    
    for (int i = 0; i < n; i += TILE_DIM) {
        for (int j = 0; j < n; j += TILE_DIM) {
            // Zero the accumulator tile for C
            _tile_zero(0);

            for (int k = 0; k < n; k += TILE_DIM) {
                // Load tiles for A and B
                _tile_loadd(1, A + i * n + k, n * sizeof(float));
                _tile_loadd(2, B + k * n + j, n * sizeof(float));
                
                // Perform matrix multiplication on the tiles
                _tile_dpbf16ps(0, 1, 2);
            }

            // Store the result back to C
            _tile_stored(0, C + i * n + j, n * sizeof(float));
        }
    }

    // Release AMX
    _tile_release();
}

int main() {
    // Allocate aligned memory for matrices A, B, C
    float *A = (float*)aligned_alloc(64, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float *B = (float*)aligned_alloc(64, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float *C = (float*)aligned_alloc(64, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize matrices A and B with some values
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
        A[i] = (float)(rand() % 100) / 100.0;
        B[i] = (float)(rand() % 100) / 100.0;
        C[i] = 0.0f;
    }

    // Perform matrix multiplication using AMX
    amx_tile_mm(A, B, C, MATRIX_SIZE);

    // Print part of the result to verify
    printf("C[0][0] = %f\n", C[0]);

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}
