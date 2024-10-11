// tiled_blur.ispc

export void tiled_blur(uniform uint8_t *in, uniform uint8_t *out, uniform int w, uniform int h, uniform int kernel_size) {
    uniform int radius = kernel_size / 2;

    // ISPC automatically parallelizes the loops across SIMD lanes
    foreach (y = 1 ... h - 1) {
        foreach (x = 1 ... w - 1) {
            foreach (c = 0 ... 3) {  // Assuming RGB channels
                int sum = 0;

                // Sum up the values within the kernel
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
