#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    long long samples = 100000000;
    if (argc > 1) {
        samples = atoll(argv[1]);
    }

    long long hits = 0;

    #pragma omp parallel
    {
        unsigned int seed = 1234u + (unsigned int)omp_get_thread_num();
        long long local_hits = 0;

        #pragma omp for
        for (long long i = 0; i < samples; i++) {
            double x = rand_r(&seed) / (double)RAND_MAX;
            double y = rand_r(&seed) / (double)RAND_MAX;
            if (x * x + y * y <= 1.0) {
                local_hits++;
            }
        }

        #pragma omp atomic
        hits += local_hits;
    }

    double pi = 4.0 * (double)hits / (double)samples;
    printf("samples=%lld hits=%lld pi=%.8f\n", samples, hits, pi);
    return 0;
}
