#include <iostream>
#include <sycl/sycl.hpp>

int main() {
    sycl::queue q;
    try {
        q = sycl::queue(sycl::gpu_selector_v);
    } catch (...) {
        q = sycl::queue(sycl::default_selector_v);
    }

    constexpr std::size_t n = 16;
    auto* data = sycl::malloc_shared<int>(n, q);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = 0;
    }

    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        data[i] = static_cast<int>(i[0] * i[0]);
    });
    q.wait();

    std::cout << "SYCL hello example\n";
    std::cout << "device=" << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "values=";
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << data[i] << (i + 1 == n ? '\n' : ' ');
    }

    sycl::free(data, q);
    return 0;
}
