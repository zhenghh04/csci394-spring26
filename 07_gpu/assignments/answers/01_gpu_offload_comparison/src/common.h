#ifndef GEMM_COMMON_H
#define GEMM_COMMON_H

#include <stddef.h>

static inline void init_matrices(float *a, float *b, float *c, int n) {
    for (size_t idx = 0; idx < (size_t)n * n; idx++) {
        a[idx] = (float)((idx % 13)) / 13.0f;
        b[idx] = (float)(((idx * 7) % 17)) / 17.0f;
        c[idx] = 0.0f;
    }
}

static inline float cpu_reference_max_err(const float *a, const float *b,
                                          const float *c, int n) {
    float max_err = 0.0f;
    int sample = n < 64 ? n : 64;
    for (int i = 0; i < sample; i++) {
        for (int j = 0; j < sample; j++) {
            float ref = 0.0f;
            for (int k = 0; k < n; k++) {
                ref += a[(size_t)i * n + k] * b[(size_t)k * n + j];
            }
            float err = ref - c[(size_t)i * n + j];
            if (err < 0.0f) err = -err;
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

#endif
