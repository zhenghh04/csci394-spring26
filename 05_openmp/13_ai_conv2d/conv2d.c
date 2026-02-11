#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static inline int idx3(int c, int h, int w, int H, int W) {
    return (c * H + h) * W + w;
}

int main(void) {
    const int C_in = 3;
    const int C_out = 16;
    const int H = 128;
    const int W = 128;
    const int K = 3;
    const int pad = 1;

    const int H_out = H;
    const int W_out = W;

    float *x = (float *)malloc((size_t)C_in * H * W * sizeof(float));
    float *w = (float *)malloc((size_t)C_out * C_in * K * K * sizeof(float));
    float *y = (float *)calloc((size_t)C_out * H_out * W_out, sizeof(float));
    if (!x || !w || !y) {
        fprintf(stderr, "Allocation failed\n");
        free(x); free(w); free(y);
        return 1;
    }

    for (int c = 0; c < C_in; c++) {
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                x[idx3(c, i, j, H, W)] = (float)((i + j + c) % 7) * 0.1f;
            }
        }
    }
    for (int oc = 0; oc < C_out; oc++) {
        for (int ic = 0; ic < C_in; ic++) {
            for (int kh = 0; kh < K; kh++) {
                for (int kw = 0; kw < K; kw++) {
                    int idx = ((oc * C_in + ic) * K + kh) * K + kw;
                    w[idx] = 0.01f * (float)((oc + ic + kh + kw) % 5 + 1);
                }
            }
        }
    }

    double t0 = omp_get_wtime();
    #pragma omp parallel for collapse(3)
    for (int oc = 0; oc < C_out; oc++) {
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                float sum = 0.0f;
                for (int ic = 0; ic < C_in; ic++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int ih = oh + kh - pad;
                            int iw = ow + kw - pad;
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                            float xv = x[idx3(ic, ih, iw, H, W)];
                            int widx = ((oc * C_in + ic) * K + kh) * K + kw;
                            sum += xv * w[widx];
                        }
                    }
                }
                y[idx3(oc, oh, ow, H_out, W_out)] = sum;
            }
        }
    }
    double t1 = omp_get_wtime();

    printf("Conv2D: C_in=%d C_out=%d H=W=%d K=%d time=%.3f s\n",
           C_in, C_out, H, K, t1 - t0);
    printf("y[0,0,0]=%.6f\n", y[idx3(0, 0, 0, H_out, W_out)]);

    free(x); free(w); free(y);
    return 0;
}
