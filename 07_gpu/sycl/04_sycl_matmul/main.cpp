#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>

int main(int argc, char** argv) {
    std::size_t n = (argc > 1) ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : 256;

    sycl::queue q;
    try {
        q = sycl::queue(sycl::gpu_selector_v);
    } catch (...) {
        q = sycl::queue(sycl::default_selector_v);
    }

    const std::size_t nn = n * n;
    float* a = sycl::malloc_shared<float>(nn, q);
    float* b = sycl::malloc_shared<float>(nn, q);
    float* c = sycl::malloc_shared<float>(nn, q);
    float* c_ref = new float[nn];

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            a[i * n + j] = static_cast<float>((i + j) % 13) / 13.0f;
            b[i * n + j] = static_cast<float>((2 * i + j) % 17) / 17.0f;
            c[i * n + j] = 0.0f;
            c_ref[i * n + j] = 0.0f;
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    q.parallel_for(sycl::range<2>(n, n), [=](sycl::id<2> idx) {
        std::size_t i = idx[0];
        std::size_t j = idx[1];
        float sum = 0.0f;
        for (std::size_t k = 0; k < n; ++k) {
            sum += a[i * n + k] * b[k * n + j];
        }
        c[i * n + j] = sum;
    });
    q.wait();
    auto t1 = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c_ref[i * n + j] = sum;
        }
    }

    double max_abs_err = 0.0;
    for (std::size_t i = 0; i < nn; ++i) {
        max_abs_err = std::max(max_abs_err, std::abs(static_cast<double>(c[i] - c_ref[i])));
    }

    double flops = 2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
    double time_s = std::chrono::duration<double>(t1 - t0).count();
    double gflops = flops / time_s / 1.0e9;

    std::cout << "SYCL matrix multiplication example\n";
    std::cout << "device=" << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "n=" << n
              << " time_s=" << time_s
              << " gflops=" << gflops
              << " max_abs_err=" << max_abs_err
              << "\n";

    delete[] c_ref;
    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    return 0;
}
