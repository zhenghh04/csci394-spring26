#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>

int main(int argc, char** argv) {
    std::size_t n = (argc > 1) ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : (1u << 20);
    float alpha = (argc > 2) ? std::strtof(argv[2], nullptr) : 2.0f;

    sycl::queue q;
    try {
        q = sycl::queue(sycl::gpu_selector_v);
    } catch (...) {
        q = sycl::queue(sycl::default_selector_v);
    }

    float* x = sycl::malloc_shared<float>(n, q);
    float* y = sycl::malloc_shared<float>(n, q);
    float* y_ref = new float[n];

    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i) / static_cast<float>(n);
        y[i] = 1.0f;
        y_ref[i] = alpha * x[i] + 1.0f;
    }

    auto t0 = std::chrono::steady_clock::now();
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        y[i] = alpha * x[i] + y[i];
    });
    q.wait();
    auto t1 = std::chrono::steady_clock::now();

    double max_abs_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        max_abs_err = std::max(max_abs_err, std::abs(static_cast<double>(y[i] - y_ref[i])));
    }

    std::cout << "SYCL AXPY example\n";
    std::cout << "device=" << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "n=" << n << " alpha=" << alpha
              << " time_s=" << std::chrono::duration<double>(t1 - t0).count()
              << " max_abs_err=" << max_abs_err << "\n";

    delete[] y_ref;
    sycl::free(x, q);
    sycl::free(y, q);
    return 0;
}
