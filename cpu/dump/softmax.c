#include <stdio.h>
#include <stdlib.h>
#include <stdfloat.h>


void softmax(float *input, float *output, int n) {
    // Find the maximum value in the input
    float max = input[0];
    for (int i = 1; i < n; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }

    // Compute the exponentials of the input values
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i] - max);
        sum += output[i];
    }

    // Normalize the output
    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

void softmax_online(const float *input, float *output, int n) {
    float max_val = -INFINITY;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float maxval_prev = max_val;
        if (inp_row[j] > max_val) {
            max_val = inp_row[j];
            sum = sum + expf(maxval_prev - max_val) + expf(inp_row[j] - max_val);
        } else {
            sum += expf(inp_row[j] - max_val);
        }
    }
    for (int i=0; i<n; i++) {
        output[i] = expf(inp_row[i] - max_val) / sum;
    }
}