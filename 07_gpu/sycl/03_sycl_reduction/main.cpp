#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>

int main(int argc, char** argv) {
    std::size_t n = (argc > 1) ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : (1u << 20);

    sycl::queue q;
    try {
        q = sycl::queue(sycl::gpu_selector_v);
    } catch (...) {
        q = sycl::queue(sycl::default_selector_v);
    }

    double* x = sycl::malloc_shared<double>(n, q);
    double* sum = sycl::malloc_shared<double>(1, q);
    *sum = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        x[i] = 1.0 / static_cast<double>(i + 1);
    }

    auto t0 = std::chrono::steady_clock::now();
    q.parallel_for(
        sycl::range<1>(n),
        sycl::reduction(sum, sycl::plus<double>()),
        [=](sycl::id<1> i, auto& acc) {
            acc += x[i];
        }
    );
    q.wait();
    auto t1 = std::chrono::steady_clock::now();

    double cpu_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        cpu_sum += x[i];
    }
    double abs_err = std::abs(*sum - cpu_sum);

    std::cout << "SYCL reduction example\n";
    std::cout << "device=" << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "n=" << n
              << " sum=" << *sum
              << " cpu_sum=" << cpu_sum
              << " abs_err=" << abs_err
              << " time_s=" << std::chrono::duration<double>(t1 - t0).count()
              << "\n";

    sycl::free(x, q);
    sycl::free(sum, q);
    return 0;
}
