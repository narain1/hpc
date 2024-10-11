#include <stdio.h>
#include <stdlib.h>

void attention_forward_cpu(float *out, float *preattn, float *attn,
                    const float *inp, int B, int T, int C, int NH){
    int C3 = C*3;
    int hs = C / NH;
    float scale = 1.0f / sqrtf(hs);

    for(int b=0; b<B; b++) {
        for (int t=0; t<T; t++) {
            for (int h=0; h<NH; h++) {
                const float *query_t = inp + b*T*C3 + t*C3 + h*hs;
                float *preattn_t = preattn +
            }
        }

    }
}