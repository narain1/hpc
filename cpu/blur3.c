#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define W 512  // Define the width and height of the image

void read_file(uint8_t *arr, char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("File opening failed");
        return;
    }
    fread(arr, sizeof(uint8_t), W * W * 3, f);
    fclose(f);
}

void write_file(uint8_t *arr, char *filename) {
    FILE *f = fopen(filename, "wb");
    fwrite(arr, sizeof(uint8_t), W * W * 3, f);
    fclose(f);
}

// ISPC tiled_blur function will be called from the ISPC-generated object file
extern void tiled_blur(uint8_t *in, uint8_t *out, int w, int h, int kernel_size);

int main() {
    // Allocate memory for the image and the output
    uint8_t *arr = (uint8_t *)malloc(W * W * 3 * sizeof(uint8_t));
    uint8_t *out = (uint8_t *)malloc(W * W * 3 * sizeof(uint8_t));

    // Read the input file into memory
    read_file(arr, "image.bin");

    // Measure the time taken for the blur function
    clock_t start = clock();
    tiled_blur(arr, out, W, W, 5);  // Example kernel size of 5
    clock_t end = clock();

    // Print the execution time
    printf("Time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Write the output image to a file
    write_file(out, "out.bin");

    // Free the allocated memory
    free(arr);
    free(out);

    return 0;
}
