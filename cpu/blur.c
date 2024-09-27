#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#define W 1024

// change operations to happen in uint8_t and 3 channels
void blur(uint8_t *in, uint8_t *out, int w, int h, int kernel_size) {
    int radius = (int)(kernel_size / 2);
    for (int c = 0; c < 3; c++) {
        for (int y = 1; y < h - 1; y++) {
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
    blur(arr, out, W, W, 5);
    clock_t end = clock();

    printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    write_file(out, "out.bin");
}