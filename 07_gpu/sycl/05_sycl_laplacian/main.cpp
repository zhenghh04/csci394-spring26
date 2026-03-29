#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>

int main(int argc, char** argv) {
    std::size_t nx = (argc > 1) ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : 1024;
    std::size_t ny = (argc > 2) ? static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10)) : 1024;

    sycl::queue q;
    try {
        q = sycl::queue(sycl::gpu_selector_v);
    } catch (...) {
        q = sycl::queue(sycl::default_selector_v);
    }

    const std::size_t n = nx * ny;
    float* u = sycl::malloc_shared<float>(n, q);
    float* out = sycl::malloc_shared<float>(n, q);
    float* ref = new float[n];

    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            std::size_t idx = j * nx + i;
            u[idx] = std::sin(0.01f * static_cast<float>(i)) * std::cos(0.02f * static_cast<float>(j));
            out[idx] = 0.0f;
            ref[idx] = 0.0f;
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    q.parallel_for(sycl::range<2>(ny, nx), [=](sycl::id<2> id) {
        std::size_t j = id[0];
        std::size_t i = id[1];
        std::size_t idx = j * nx + i;

        if (i == 0 || j == 0 || i + 1 == nx || j + 1 == ny) {
            out[idx] = 0.0f;
        } else {
            out[idx] =
                -4.0f * u[idx] +
                u[j * nx + (i - 1)] +
                u[j * nx + (i + 1)] +
                u[(j - 1) * nx + i] +
                u[(j + 1) * nx + i];
        }
    });
    q.wait();
    auto t1 = std::chrono::steady_clock::now();

    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            std::size_t idx = j * nx + i;
            if (i == 0 || j == 0 || i + 1 == nx || j + 1 == ny) {
                ref[idx] = 0.0f;
            } else {
                ref[idx] =
                    -4.0f * u[idx] +
                    u[j * nx + (i - 1)] +
                    u[j * nx + (i + 1)] +
                    u[(j - 1) * nx + i] +
                    u[(j + 1) * nx + i];
            }
        }
    }

    double max_abs_err = 0.0;
    for (std::size_t idx = 0; idx < n; ++idx) {
        max_abs_err = std::max(max_abs_err, std::abs(static_cast<double>(out[idx] - ref[idx])));
    }

    std::cout << "SYCL Laplacian example\n";
    std::cout << "device=" << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "nx=" << nx
              << " ny=" << ny
              << " time_s=" << std::chrono::duration<double>(t1 - t0).count()
              << " max_abs_err=" << max_abs_err
              << "\n";

    delete[] ref;
    sycl::free(u, q);
    sycl::free(out, q);
    return 0;
}
