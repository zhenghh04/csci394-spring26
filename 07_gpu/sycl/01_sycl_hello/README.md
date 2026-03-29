# SYCL Hello

This example performs a tiny kernel launch with SYCL and prints the selected
device.

## Self-explanation

This is the smallest SYCL example in this folder.

The code does four things:

1. creates a `sycl::queue`
2. tries to select a GPU device first
3. allocates a small shared array
4. launches a `parallel_for` kernel that fills the array

Why this example matters:

- it shows the basic structure of a SYCL program
- it shows how host code submits work to a device queue
- it shows that `malloc_shared` lets both the CPU and GPU see the same memory
- it gives a quick sanity check that the compiler and runtime are working

How to read the code:

- `sycl::queue q(...)`
  - chooses the device and manages kernel submission
- `sycl::malloc_shared`
  - allocates unified shared memory accessible by both host and device
- `q.parallel_for(...)`
  - launches one work-item per array element
- `q.wait()`
  - waits until the device work is complete before printing results

Concepts:

- queue creation
- GPU selection
- `parallel_for`
- basic device synchronization

Build:

```bash
make
```

Run:

```bash
./app
```

Expected behavior:

- the program prints the selected device name
- it prints a short list of values computed on the device
