#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define W 1024
#define XS 64
#define TILE_SIZE 32

// change operations to happen in uint8_t and 3 channels
void blur(uint8_t *in, uint8_t *out, int w, int h, int kernel_size) {
    int radius = (int)(kernel_size / 2);

    // Parallelize the outer loop using OpenMP
    #pragma omp parallel for collapse(2)
    for (int c = 0; c < 3; c++) { // Iterate over the channels
        for (int y = 1; y < h - 1; y++) { // Parallelize the rows
            for (int x = 1; x < w - 1; x++) {
                int sum = 0;

                for (int j = -radius; j <= radius; j++) {
                    for (int i = -radius; i <= radius; i++) {
                        sum += in[(y + j) * w * 3 + (x + i) * 3 + c];
                    }
                }

                out[y * w * 3 + x * 3 + c] = (uint8_t)(sum / (kernel_size * kernel_size));
            }
        }
    }
}

void blur2(uint8_t *in, uint8_t *out, int w, int h, int kernel_size) {
    int radius = kernel_size / 2;

    for (int c = 0; c < 3; c++) { // Iterate over the channels
        for (int start_col = 0; start_col < w; start_col += XS) {
            int end_col = (start_col + XS < w) ? start_col + XS : w;

            for (int y = 1; y < h - 1; y++) {
                for (int x = start_col; x < end_col; x++) { // Process 32 columns at a time
                    int sum = 0;
                    
                    for (int j = -radius; j <= radius; j++) {
                        for (int i = -radius; i <= radius; i++) {
                            int yy = y + j;
                            int xx = x + i;
                            
                            // Make sure we don't go out of bounds
                            if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
                                sum += in[yy * w * 3 + xx * 3 + c];
                            }
                        }
                    }
                    
                    // Assign the blurred value to the output array
                    out[y * w * 3 + x * 3 + c] = (uint8_t)(sum / (kernel_size * kernel_size));
                }
            }
        }
    }
}

void tiled_blur(uint8_t *in, uint8_t *out, int w, int h, int kernel_size) {
    int radius = kernel_size / 2;

    // Parallelize using OpenMP
    #pragma omp parallel for collapse(2)
    for (int c = 0; c < 3; c++) { // Iterate over the color channels
        for (int ty = 0; ty < h; ty += TILE_SIZE) { // Iterate over tiles vertically
            for (int tx = 0; tx < w; tx += TILE_SIZE) { // Iterate over tiles horizontally
                // Process each tile (32x32 block)
                for (int y = ty; y < ty + TILE_SIZE && y < h; y++) {
                    for (int x = tx; x < tx + TILE_SIZE && x < w; x++) {
                        int sum = 0;
                        
                        // Apply blur kernel to the current pixel
                        for (int j = -radius; j <= radius; j++) {
                            for (int i = -radius; i <= radius; i++) {
                                int yy = y + j;
                                int xx = x + i;

                                // Ensure we don't go out of bounds
                                if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
                                    sum += in[yy * w * 3 + xx * 3 + c];
                                }
                            }
                        }
                        
                        // Assign the blurred value to the output
                        out[y * w * 3 + x * 3 + c] = (uint8_t)(sum / (kernel_size * kernel_size));
                    }
                }
            }
        }
    }
}

void read_file(uint8_t *arr, char *filename) {
    FILE *f = fopen(filename, "rb");
    fread(arr, sizeof(uint8_t), W * W * 3, f);
    fclose(f);
}

void write_file(uint8_t *arr, char *filename) {
    FILE *f = fopen(filename, "wb");
    fwrite(arr, sizeof(uint8_t), W * W * 3, f);
    fclose(f);
}

int main() {
    uint8_t *arr = (uint8_t*)malloc(W * W * 3 * sizeof(uint8_t));
    uint8_t *out = (uint8_t*)malloc(W * W * 3 * sizeof(uint8_t));
    read_file(arr, "image.bin");
    // time blur
    clock_t start = clock();
    tiled_blur(arr, out, W, W, 5);
    clock_t end = clock();

    printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    write_file(out, "out.bin");
}