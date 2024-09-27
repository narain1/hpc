#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#define W 1024

struct ImgU8 {
    uint8_t *data;
    int w, h;
    int start, end;
};

// has non contiguous memory access
void blur(uint8_t *in, uint8_t *out, int w, int h, int kernel_size) {
    int radius = (int)(kernel_size / 2);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int sum = 0;
                int count = 0;
                for (int j = -radius; j <= radius; j++) {
                    for (int i = -radius; i <= radius; i++) {
                        int x2 = x + i;
                        int y2 = y + j;
                        if (x2 >= 0 && x2 < w && y2 >= 0 && y2 < h) {
                            sum += in[(y2 * w + x2) * 3 + c];
                            count++;
                        }
                    }
                }
                out[(y * w + x) * 3 + c] = (uint8_t)(sum / count);
            }
        } 
    }
}

void img2col(uint8_t *in, int h, int w, int c, int k, int s, uint8_t *col_matrix) {
    for (int y=0; y <= h - k; y += s) {
        for (int x=0; x <= w - k; x += s) {
            for (int i=0; i < k; i++) {
                for (int j=0; j < k; j++) {
                    int input_idx = :w
                    col_matrix[i * j * c + col_idx] = in[(y + i) * w + x + j];
                }
            }
        }
    }
}

void blur2(uint8_t *in, uint8_t *out, int w, int h, int k) {
    // create a matrix 
    uint8_t *col

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
    clock_t start = clock();
    blur(arr, out, W, W, 10);
    clock_t end = clock();

    printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    write_file(out, "out.bin");
}