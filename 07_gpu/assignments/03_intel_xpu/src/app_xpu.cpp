// Intel XPU GEMM benchmark on Aurora.
// Two paths:
//   fp32           — naive SYCL FP32 GEMM kernel (USM device memory)
//   xmx_bf16_fp32  — joint_matrix tiled GEMM (BF16 in, FP32 accum)
//
// Usage: ./app_xpu <mode> <n> <iters> <warmup>
// Output:
//   RESULT,<mode>,n,iters,warmup,time_s,gflops,max_abs_err
//
// Build: icpx -fsycl -O2 -fsycl-targets=spir64_gen -Xs "-device pvc" \
//        -o app_xpu app_xpu.cpp

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace ext_matrix = sycl::ext::oneapi::experimental::matrix;
using bfloat16 = sycl::ext::oneapi::bfloat16;

static double wclock(void) {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

static void init_host(float *a, float *b, int n) {
    for (size_t idx = 0; idx < (size_t)n * n; idx++) {
        a[idx] = (float)((idx % 13)) / 13.0f;
        b[idx] = (float)(((idx * 7) % 17)) / 17.0f;
    }
}

static float ref_max_err_block(const float *a, const float *b, const float *c,
                               int n, int sample) {
    float max_err = 0.0f;
    for (int i = 0; i < sample; i++) {
        for (int j = 0; j < sample; j++) {
            float s = 0.0f;
            for (int k = 0; k < n; k++) s += a[i*n + k] * b[k*n + j];
            float e = std::fabs(s - c[i*n + j]);
            if (e > max_err) max_err = e;
        }
    }
    return max_err;
}

// Naive FP32 SYCL GEMM (row-major, A,B,C all n x n)
static void run_fp32(sycl::queue &q, const float *dA, const float *dB,
                     float *dC, int n) {
    q.parallel_for(sycl::range<2>(n, n), [=](sycl::id<2> idx) {
        int i = idx[0], j = idx[1];
        float s = 0.0f;
        for (int k = 0; k < n; k++) s += dA[i * n + k] * dB[k * n + j];
        dC[i * n + j] = s;
    }).wait();
}

// joint_matrix tile sizes (PVC supports BF16 8x16x16)
constexpr int TM = 8;
constexpr int TN = 16;
constexpr int TK = 16;
constexpr int SG_SIZE = 16;

static void run_xmx_bf16(sycl::queue &q, const bfloat16 *dA, const bfloat16 *dB,
                         float *dC, int n) {
    if ((n % TM) || (n % TN) || (n % TK)) {
        std::fprintf(stderr, "warning: n=%d not divisible by TM/TN/TK; xmx path skipped\n", n);
        return;
    }
    int m_tiles = n / TM;
    int n_tiles = n / TN;

    sycl::range<2> global(m_tiles, n_tiles * SG_SIZE);
    sycl::range<2> local(1, SG_SIZE);

    q.submit([&](sycl::handler &h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();
                int row_tile = item.get_group(0);
                int col_tile = item.get_group(1);

                ext_matrix::joint_matrix<sycl::sub_group, float,
                                         ext_matrix::use::accumulator,
                                         TM, TN> tC;
                ext_matrix::joint_matrix_fill(sg, tC, 0.0f);

                for (int kt = 0; kt < n / TK; kt++) {
                    ext_matrix::joint_matrix<sycl::sub_group, bfloat16,
                                             ext_matrix::use::a,
                                             TM, TK,
                                             ext_matrix::layout::row_major> tA;
                    ext_matrix::joint_matrix<sycl::sub_group, bfloat16,
                                             ext_matrix::use::b,
                                             TK, TN,
                                             ext_matrix::layout::row_major> tB;

                    const bfloat16 *Aptr = dA + (row_tile * TM) * n + kt * TK;
                    const bfloat16 *Bptr = dB + (kt * TK) * n + col_tile * TN;

                    ext_matrix::joint_matrix_load(sg, tA,
                        sycl::address_space_cast<sycl::access::address_space::global_space,
                                                 sycl::access::decorated::no>(
                            const_cast<bfloat16 *>(Aptr)),
                        n);
                    ext_matrix::joint_matrix_load(sg, tB,
                        sycl::address_space_cast<sycl::access::address_space::global_space,
                                                 sycl::access::decorated::no>(
                            const_cast<bfloat16 *>(Bptr)),
                        n);
                    ext_matrix::joint_matrix_mad(sg, tC, tA, tB, tC);
                }
                float *Cptr = dC + (row_tile * TM) * n + col_tile * TN;
                ext_matrix::joint_matrix_store(sg, tC,
                    sycl::address_space_cast<sycl::access::address_space::global_space,
                                             sycl::access::decorated::no>(Cptr),
                    n, ext_matrix::layout::row_major);
            });
    }).wait();
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s {fp32|xmx_bf16_fp32} n [iters=5] [warmup=1]\n", argv[0]);
        return 1;
    }
    std::string mode = argv[1];
    int n = std::atoi(argv[2]);
    int iters = (argc > 3) ? std::atoi(argv[3]) : 5;
    int warmup = (argc > 4) ? std::atoi(argv[4]) : 1;

    sycl::queue q;
    try {
        q = sycl::queue(sycl::gpu_selector_v);
    } catch (...) {
        q = sycl::queue(sycl::default_selector_v);
    }
    std::fprintf(stderr, "device: %s\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str());

    size_t nn = (size_t)n * n;
    std::vector<float> hA(nn), hB(nn), hC(nn, 0.0f);
    init_host(hA.data(), hB.data(), n);

    float *dA_f32 = sycl::malloc_device<float>(nn, q);
    float *dB_f32 = sycl::malloc_device<float>(nn, q);
    float *dC_f32 = sycl::malloc_device<float>(nn, q);
    q.memcpy(dA_f32, hA.data(), nn * sizeof(float)).wait();
    q.memcpy(dB_f32, hB.data(), nn * sizeof(float)).wait();
    q.memset(dC_f32, 0, nn * sizeof(float)).wait();

    bfloat16 *dA_bf = nullptr;
    bfloat16 *dB_bf = nullptr;
    if (mode == "xmx_bf16_fp32") {
        dA_bf = sycl::malloc_device<bfloat16>(nn, q);
        dB_bf = sycl::malloc_device<bfloat16>(nn, q);
        std::vector<bfloat16> tmpA(nn), tmpB(nn);
        for (size_t i = 0; i < nn; i++) {
            tmpA[i] = bfloat16(hA[i]);
            tmpB[i] = bfloat16(hB[i]);
        }
        q.memcpy(dA_bf, tmpA.data(), nn * sizeof(bfloat16)).wait();
        q.memcpy(dB_bf, tmpB.data(), nn * sizeof(bfloat16)).wait();
    }

    auto launch = [&]() {
        if (mode == "fp32") {
            run_fp32(q, dA_f32, dB_f32, dC_f32, n);
        } else {
            run_xmx_bf16(q, dA_bf, dB_bf, dC_f32, n);
        }
    };

    for (int w = 0; w < warmup; w++) launch();

    double t0 = wclock();
    for (int it = 0; it < iters; it++) launch();
    q.wait();
    double t_total = wclock() - t0;
    double per_call = t_total / iters;

    q.memcpy(hC.data(), dC_f32, nn * sizeof(float)).wait();

    int sample = n < 64 ? n : 64;
    float err = ref_max_err_block(hA.data(), hB.data(), hC.data(), n, sample);

    double gflops = 2.0 * (double)n * n * n / per_call / 1.0e9;
    std::printf("RESULT,%s,%d,%d,%d,%.6e,%.3f,%.3e\n",
                mode.c_str(), n, iters, warmup, per_call, gflops, err);

    sycl::free(dA_f32, q);
    sycl::free(dB_f32, q);
    sycl::free(dC_f32, q);
    if (dA_bf) sycl::free(dA_bf, q);
    if (dB_bf) sycl::free(dB_bf, q);
    return 0;
}
